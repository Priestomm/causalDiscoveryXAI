import argparse 

from causal_discovery import *

## for parsing options in terminal 
def parse_option():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['PCMCI'], default='PCMCI')
    parser.add_argument('--dataset_name', choices=['boat', 'pepper', 'swat'], default='pepper', help='Select the dataset: boat, pepper or swat')
    parser.add_argument('--assumption', choices=['linear', 'parametric', 'not_linear'], default='parametric', help='Choose between parcorr, gpdc and knn')

    return parser.parse_args()

## run the main algorithm 
def run(*kwargs):

    opt = parse_option()

    print('-'*50)
    print(f'Anomaly detection with causal discovery...\nmode: {opt.mode}\ndataset: {opt.dataset_name}\nassumptions: {opt.assumption}')
    print('-'*50)

    # get the data 
    normal_df = read_preprocess_data(opt.dataset_name, 'normal', SUBSET_SIZE)
    attack_df = read_preprocess_data(opt.dataset_name, 'attack', SUBSET_SIZE)
    
    # remove nan 
    normal_df = normal_df.fillna(0)
    attack_df = attack_df.fillna(0)

    # # remove 200 columns 
    # normal_df = normal_df.iloc[:, :-200]
    # attack_df = attack_df.iloc[:, :-200]

    # get var names 
    var_names_normal = list(normal_df.columns)
    var_names_attack = list(attack_df.columns)

    # convert in dataframe
    normal_data = pp.DataFrame(normal_df.values, var_names=var_names_normal)
    attack_data = pp.DataFrame(attack_df.values, var_names=var_names_attack)

    discovery(normal_data, SUBSET_SIZE, 'normal', var_names_normal, opt.dataset_name, opt.assumption, opt.mode)
    discovery(attack_data, SUBSET_SIZE, 'attack', var_names_attack, opt.dataset_name, opt.assumption, opt.mode)


if __name__ == '__main__':

    opt = parse_option()
    run(opt)