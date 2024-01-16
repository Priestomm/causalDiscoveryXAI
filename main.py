import argparse 

from causal_discovery import *

## for parsing options in terminal 
def parse_option():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['PCMCI'], default='PCMCI')
    parser.add_argument('--dataset_name', choices=['boat', 'pepper', 'swat'], default='pepper', help='Select the dataset: boat, pepper or swat')
    parser.add_argument('--assumption', choices=['linear', 'not_linear'], default='not_linear', help='Choose between parcorr and knn')

    return parser.parse_args()

## run the main algorithm 
def run(*kwargs):

    opt = parse_option()

    print('-'*50)
    print(f'Anomaly detection with causal discovery...\nmode: {opt.mode}\ndataset: {opt.dataset_name}\nassumptions: {opt.assumption}')
    print('-'*50)

    # get the data 
    normal_df = read_preprocess_data(opt.dataset_name, 'normal').fillna(0)

    if opt.dataset_name == 'boat':
        gpsdown_df = read_preprocess_data(opt.dataset_name, 'fault_gpsdown').fillna(0)
        stucked_df = read_preprocess_data(opt.dataset_name, 'fault_stucked').fillna(0)
    elif opt.dataset_name == 'pepper':
        wheels_df = read_preprocess_data(opt.dataset_name, 'WheelsControl').fillna(0)
        joint_df = read_preprocess_data(opt.dataset_name, 'JointControl').fillna(0)
    elif opt.dataset_name == 'swat':
        attack_df = read_preprocess_data(opt.dataset_name, 'attack').fillna(0)
    
    # get var names 
    var_names_normal = list(normal_df.columns)

    if opt.dataset_name == 'boat':
        var_names_gpsdown = list(gpsdown_df.columns)
        var_names_stucked = list(stucked_df.columns)
    elif opt.dataset_name == 'pepper':
        var_names_wheels = list(wheels_df.columns)
        var_names_joint = list(joint_df.columns)
    elif opt.dataset_name == 'swat':
        var_names_attack = list(attack_df.columns)

    # convert in dataframe
    normal_data = pp.DataFrame(normal_df.values, var_names=var_names_normal)

    if opt.dataset_name == 'boat':
        gpsdown_data = pp.DataFrame(gpsdown_df.values, var_names=var_names_gpsdown)
        stucked_data = pp.DataFrame(stucked_df.values, var_names=var_names_stucked)
    elif opt.dataset_name == 'pepper':
        wheels_data = pp.DataFrame(wheels_df.values, var_names=var_names_wheels)
        joint_data = pp.DataFrame(joint_df.values, var_names=var_names_joint)
    elif opt.dataset_name == 'swat':
        attack_data = pp.DataFrame(attack_df.values, var_names=var_names_attack)

    discovery(normal_data, SUBSET_SIZE, 'normal', var_names_normal, opt.dataset_name, opt.assumption, opt.mode)
    if opt.dataset_name == 'boat':
        discovery(gpsdown_data, SUBSET_SIZE, 'gpsdown', var_names_gpsdown, opt.dataset_name, opt.assumption, opt.mode)
        discovery(stucked_data, SUBSET_SIZE, 'stucked', var_names_stucked, opt.dataset_name, opt.assumption, opt.mode)
    elif opt.dataset_name == 'pepper': 
        discovery(wheels_data, SUBSET_SIZE, 'wheels', var_names_wheels, opt.dataset_name, opt.assumption, opt.mode)
        discovery(joint_data, SUBSET_SIZE, 'joint', var_names_joint, opt.dataset_name, opt.assumption, opt.mode)
    elif opt.dataset_name == 'swat':
        discovery(attack_data, SUBSET_SIZE, 'attack', var_names_attack, opt.dataset_name, opt.assumption, opt.mode)


if __name__ == '__main__':

    opt = parse_option()
    run(opt)