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
from tigramite.independence_tests.gpdc import GPDC

from sklearn.gaussian_process.kernels import (RBF, ExpSineSquared, 
                                              RationalQuadratic, WhiteKernel)
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import numpy as np


from scipy.stats import ConstantInputWarning
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from zmq import SUB
warnings.filterwarnings('ignore', category=ConstantInputWarning)

SUBSET_SIZE = 1000
#ALPHA = 0.00001
ALPHA = 0.05
PC_ALPHA = 0.05 # variare per knn 0.01-0.05
TAU_MAX = 2
VERBOSITY = 0

## read and preprocesses data for each dataset
def read_preprocess_data(dataset_name, dataset_type, SUBSET_SIZE=SUBSET_SIZE):

	if dataset_name == 'pepper':

		df = pd.read_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', delimiter=',')
		df = df.drop(index=0) # Remove first row
		df = df.loc[:, (df != df.iloc[0]).any()] # Remove near constant columns
		df = df.dropna(axis=1, how='all') # Drop columns that are entirely NaN
		df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # Convert timestamp to datetime
		df.set_index('timestamp', inplace=True) # Set timestamp as index
		columns_to_drop = ['timestamp']
		df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

		# cast all cell values to float
		df = df.apply(pd.to_numeric, errors='coerce')

		# Variance for each column and keep the 50 largest
		variances = df.var().nlargest(50)
		df = df[variances.index]

		df = sample_rows(df, 100)
		
		return df

	elif dataset_name == 'boat':
		df = pd.read_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', delimiter=',')
		
		# remove first row if it's a string
		if isinstance(df.iloc[0, 0], str):
			idxs = []
			for idx, type_ in enumerate(df.iloc[0, :]):
				if type_ == "String":
					idxs.append(idx)

			df = df.drop(df.columns[idxs], axis=1) # Remove string columns	
			df = df.drop(index=0) # Remove first row
			df.to_csv(f'./{dataset_name}_dataset/{dataset_type}.csv', index=False)
		
		df = df.dropna(axis=1, how='all')

		if dataset_type == 'attack':
			df['net_rec_tstamp'] = pd.to_datetime(df['net_rec_tstamp'], unit='ms')
		if dataset_type == 'normal':
			df['net_send_tstamp'] = pd.to_datetime(df['net_send_tstamp'], unit='ms')

		# Cast all cell values to float
		df = df.apply(pd.to_numeric, errors='coerce')

		df = sample_rows(df, 50)


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

		df = sample_rows(df, 50)

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
	
	elif assumptions == 'parametric':
		
		start_time = time.time()
		#noise_kernel = 0.1**2 * RBF(length_scale=0.1, length_scale_bounds=(1e-2,1e7)) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))
		#pcmci = PCMCI(dataframe=data, cond_ind_test=GPDC(significance='analytic', gp_params={"kernel":noise_kernel}), verbosity=VERBOSITY)
		pcmci = PCMCI(dataframe=data, cond_ind_test=GPDC(significance='analytic', gp_params=None), verbosity=VERBOSITY)
		results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA)
		end_time = time.time()
		print("Time taken for PMCI with subset size", subset_size, "-", round(end_time - start_time, 2), "seconds")

		return results


	elif assumptions == 'not_linear':

		start_time = time.time()
		pcmci = PCMCI(dataframe=data, cond_ind_test=CMIknn(significance="fixed_thres", knn=5))
		results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA)
		end_time = time.time()
		print("Time taken for PMCI with subset size", subset_size, "-", round(end_time - start_time, 2), "seconds")

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
	
	if results != None:
		print("Performing plot_and_extract_links function...")
		plot_and_extract_links(results, dataset_type, var_names, dataset_name, assumption, mode)
	else:
		print("Results are None...")

def sample_rows(df, freq_rate):
    freq_cols = {}
    top_components = 5

    # Apply Fourier transform to each column
    for col_name in df.columns:
        column_data = df[col_name].values

        # Calculate the Fourier transform
        fft_result = fft(column_data)
        frequencies = fftfreq(len(fft_result))

        # Find the peaks in the Fourier transform
        peaks, _ = find_peaks(np.abs(fft_result), height=10)

        # Select the top 5 peaks (or fewer, if there are fewer than 5 peaks)
        selected_peaks = peaks[np.argsort(np.abs(fft_result[peaks]))[::-1][:top_components]]

        # Retrieve the frequencies and amplitudes of the selected peaks
        selected_frequencies = frequencies[selected_peaks]
        selected_amplitudes = np.abs(fft_result[selected_peaks])
        freq_cols[col_name] = (selected_frequencies, selected_amplitudes)

    # Take the maximum among all frequencies
    max_freq = 0
    for col_name in freq_cols:
        if (len(freq_cols[col_name][0]) == 0):
            continue
        freqs, amps = freq_cols[col_name]
        max_freq = max(max_freq, max(freqs))

    sr = max_freq * freq_rate

    # Sample the rows of df with a sampling frequency of sr
    df = df.iloc[::int(sr), :]

    return df