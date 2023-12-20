import os
import matplotlib.pyplot as plt 
import time 
import pandas as pd 
import numpy as np 
import warnings
import re
import tigramite
import sys

from sklearn.feature_selection import VarianceThreshold 
from typing import TypeAlias

from tigramite import data_processing as pp
from tigramite import plotting as tp 
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn


from scipy.stats import ConstantInputWarning
from zmq import SUB
warnings.filterwarnings('ignore', category=ConstantInputWarning)

SUBSET_SIZE = 1000
ALPHA = 0.05
PC_ALPHA = 0.05
TAU_MAX = 2
VERBOSITY = 0

## read and preprocesses data for each dataset
def read_preprocess_data(dataset_name, dataset_type, SUBSET_SIZE=SUBSET_SIZE):

	if dataset_name == 'pepper':

		df = pd.read_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', delimiter=',')
		df = df.drop(index=0) # Remove first row
		df = df.tail(SUBSET_SIZE)
		df = df.loc[:, (df != df.iloc[0]).any()]
		df = df.dropna(axis=1, how='all')
		df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
		df.set_index('timestamp', inplace=True)
		columns_to_drop = ['timestamp']
		df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
		
		
		return df

	elif dataset_name == 'boat':
		## remove dumbas type columns
		df = pd.read_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', delimiter=',')
		df = df.drop(index=0) # Remove first row
		df.to_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', index=False)

		with open('./boat_dataset/types.txt', 'r') as f:
			lines = f.readlines()

		types_dict = {}
		for line in lines:
			key, value = line.strip().split()
			types_dict[key] = value

		df = pd.read_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', delimiter=',', dtype=types_dict)

		types = types_dict.values()
    	
		idxs = []
		for idx, type_ in enumerate(types):
			if type_ == "String":
				idxs.append(idx)


		df = df.drop(df.columns[idxs], axis=1) # Remove string columns		
		# df = df.drop(index=0) # Remove first row
		
		df = df.tail(SUBSET_SIZE)
		# df = df.loc[:, (df != df.iloc[0]).any()]
		df = df.dropna(axis=1, how='all')

		if dataset_type == 'attack':
			df['net_rec_tstamp'] = pd.to_datetime(df['net_rec_tstamp'], unit='ms')
			df['net_send_tstamp'] = pd.to_datetime(df['net_send_tstamp'], unit='ms')
		if dataset_type == 'normal':
			df['net_send_tstamp'] = pd.to_datetime(df['net_send_tstamp'], unit='ms')

		return df

	elif dataset_name == 'swat':

		df = pd.read_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', delimiter=';')
		df = df.tail(SUBSET_SIZE)
		df = df.loc[:, (df != df.iloc[0]).any()] # Remove near constant columns
		df = df.dropna(axis=1, how='all') # Drop columns that are entirely NaN
		df['Timestamp'] = df[' Timestamp'].str.strip()
		df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p")
		df.set_index('Timestamp', inplace=True)
		columns_to_drop = [' Timestamp', 'Normal/Attack']
		df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

		return df

## performs pcmci algorithm with parcorr
def run_pcmci(data, subset_size, assumptions):

	if assumptions == 'linear':

		start_time = time.time()
		pcmci = PCMCI(dataframe=data, cond_ind_test=ParCorr(), verbosity=VERBOSITY)
		results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA)
		end_time = time.time()
		print("Time taken for PMCI with subset size", subset_size, "-", round(end_time - start_time, 2), "seconds")

		return results

	elif assumptions == 'not_linear':

		start_time = time.time()
		pcmci = PCMCI(dataframe=data, cond_ind_test=CMIknn(knn=5))
		results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA)
		end_time = time.time()
		print("------> Time taken for PMCI with subset size", subset_size, "-", round(end_time - start_time, 2), "seconds")

		return results

## performs lpcmci algorithm with parcorr
def run_lpcmci(data, subset_size, assumption, var_names):

	if assumption == 'linear':

		start_time = time.time()
		lpcmci = LPCMCI(data, cond_ind_test=ParCorr())

		## link assumptions
		link_assumptions = {}
		for j in range(len(var_names)):
			if j in var_names:
				link_assumptions[j] = {(var, -lag): '-?>' for var in var_names for lag in range(1, TAU_MAX)}
				link_assumptions[j].update({(var, 0): 'o?o' for var in var_names if var != j})
			else:
				link_assumptions[j] = {}

		results = lpcmci.run_lpcmci(tau_max=TAU_MAX, link_assumptions=link_assumptions, pc_alpha=PC_ALPHA) ## link assumptions?
		end_time = time.time()
		print("------> Time taken for PMCI with subset size", subset_size, "-", round(end_time - start_time, 2))

		return results 
	
	elif assumption == 'not_linear':

		start_time = time.time()
		lpcmci = LPCMCI(data, cond_ind_test=CMIknn(knn=5))
		results = lpcmci.run_lpcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA) ## link assumptions?
		end_time = time.time()
		print("------> Time taken for PMCI with subset size", subset_size, "-", round(end_time - start_time, 2))
		
		return results


## plots/saves graphs type_ = normal/attack, mode = LPCMI/PCMCI
def plot_and_extract_links(results, graph_name_prefix, var_names, dataset_name, type_, mode):
	# if ./images_result/ doesn't exist, create it
	try:
		os.mkdir('./images_result')
	except OSError as error:
		pass
	tp.plot_graph(
		val_matrix=results['val_matrix'],
		graph=results['graph'],
		var_names=var_names,
		link_colorbar_label='cross-MCI',
		node_colorbar_label='auto-MCI',
		show_autodependency_lags=False,
		figsize=(20, 20),
		node_size=0.1,
		arrow_linewidth=3
	); plt.title(graph_name_prefix); plt.savefig(f'./images_result/{dataset_name}_{graph_name_prefix}_{type_}_{mode}.png')
	print(f"Image saved! --> ./images_result/{dataset_name}_{graph_name_prefix}_{type_}_{mode}.png")

## performs the discovery algorithms dataset_type = 'normal/attack'
def discovery(data, SUBSET_SIZE, dataset_type, var_names, dataset_name, assumption, mode):

	results = None
	print(f"\nPerforming {mode} algorithm...\n")

	if mode == 'PCMCI':
		results = run_pcmci(data, SUBSET_SIZE, assumption)
	elif mode == 'LPCMCI':
		results = run_lpcmci(data, SUBSET_SIZE, assumption, var_names)
	
	if results != None:
		print("Performing plot_and_extract_links function...")
		plot_and_extract_links(results, dataset_type, var_names, dataset_name, assumption, mode)
	else:
		print("Results are None...")

