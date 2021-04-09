import matplotlib as mpl
mpl.use('Agg')

from os.path import join, exists, basename, dirname, splitext
import os
import numpy as np
import argparse
from glob import glob
import time
import random
from data_augmentation import DataAugmenter
from train_classifier import train_classifier

def launch_experiment(dataset_dir, output_dir, experiment_tag, augmentation_tag, organ_tag, network_tag, trial_tag,
                      workers, batch_size, lr, n_epochs, test_only=False, ignore_std_test_set=False, prob_white_patch=None,randaugment=True, rand_m=5, rand_n=1,ra_type='Default', v1_type=None, v2_type=None, t1_type=None, t2_type=None):



    print("In launch_experiment")
    # from utils import dl

    # Color space
    if experiment_tag == 'grayscale':
        color_space = 'grayscale'
        patch_size = (128, 128, 1)
    else:
        color_space = 'rgb'
        patch_size = (128, 128, 3)
    
    # Prepare directories
    if randaugment:
        # Prepare directories
        
        classifier_dir = join(output_dir, v1_type+'_'+v2_type+'_'+t1_type+'_'+t2_type,experiment_tag, augmentation_tag, organ_tag, network_tag,'n_'+str(rand_n),'m_'+str(rand_m), trial_tag)
        
    else:
        classifier_dir = join(output_dir, experiment_tag, augmentation_tag, organ_tag, network_tag, trial_tag)
    if not exists(classifier_dir):
        os.makedirs(classifier_dir)
    print('ra_type',ra_type)
    print('randaugment:',randaugment)
    print('v1_type,v2_type, t1_type, t2_type:',v1_type,v2_type, t1_type, t2_type)
    # Prepare data
    if experiment_tag == 'std':
        patches_folder = 'patches_std'

    elif 'network-std' in experiment_tag:
        patches_folder = 'patches_' + experiment_tag

    elif 'bugetal' == experiment_tag:
        patches_folder = 'patches_' + experiment_tag

    elif 'macenko' == experiment_tag:
        patches_folder = 'patches_' + experiment_tag

    else:
        patches_folder = 'patches'

    # Data dirs
    organ_dir = join(dataset_dir, organ_tag, patches_folder)
    if ignore_std_test_set:
        test_dataset_dir = join(dataset_dir, organ_tag, 'patches')
    else:
        test_dataset_dir = None

    # Train model
    train_classifier(
            dataset_dir=organ_dir,
            output_dir=classifier_dir,
            n_epochs=n_epochs,
            batch_size=batch_size,
            augmenter=DataAugmenter(augmentation_tag=augmentation_tag),
            lr=lr,
            patch_size=patch_size,
            workers=workers,
            network_tag=network_tag,
            color_space=color_space,
            test_only=test_only,
            test_dataset_dir=test_dataset_dir,
            randaugment=randaugment, 
            rand_m=rand_m,  
            rand_n=rand_n,
            ra_type=ra_type,
            v1_type=v1_type,
            v2_type=v2_type,
            t1_type=t1_type,
            t2_type=t2_type,
        )


#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.
    """

    # Configure argument parser.
    #
    print("In collect_arguments")
    argument_parser = argparse.ArgumentParser(description='Trains a classifier given experiment parameters.')
    
    argument_parser.add_argument('-d', '--dataset_dir', required=False,       type=str, default="data",   help='directory where the data lives')
    
    argument_parser.add_argument('-o', '--output_dir',  required=False,       type=str, default="randaugment_multi_hsv_randomize_hed/models",  help='directory where the results will be stored')
    argument_parser.add_argument('-et', '--experiment_tag',         required=False,       type=str, default="rgb",  help='experiment identifier')
    argument_parser.add_argument('-at', '--augmentation_tag',         required=False,       type=str, default="baseline",   help='augmentation identifier')
    argument_parser.add_argument('-ot', '--organ_tag',         required=False,       type=str, default="lymph",   help='organ identifier')
    argument_parser.add_argument('-nt', '--network_tag',         required=False,       type=str, default="basic",   help='network identifier')
    argument_parser.add_argument('-tt', '--trial_tag',         required=False,       type=str, default="1",   help='trial identifier')
    argument_parser.add_argument('-n', '--n_epochs',     required=False,       type=int, default=120,    help='number of training epochs to run')
    argument_parser.add_argument('-b', '--batch_size',   required=False,       type=int, default=64,   help='number of samples per batch')
    argument_parser.add_argument('-l', '--lr',           required=False,       type=float, default=0.003,  help='learning_rate')
    argument_parser.add_argument('-w',  '--workers',        required=False,      type=int,   default=1, help='number of parallel workers to use')
    argument_parser.add_argument('-to', '--test_only',         required=False,       type=str, default='False',  help='test only mode')
    argument_parser.add_argument('-ist', '--ignore_std_test_set',         required=False,       type=str, default='False',  help='do not standardize the test set during test phase')
    argument_parser.add_argument('-pwp', '--prob_white_patch',         required=False,       type=str, default='None',  help='probability of including a white patch in the batch')
    argument_parser.add_argument('-rand', '--randaugment',         required=False,       type=str, default='True',  help='apply randaugment')
    argument_parser.add_argument('-rm', '--rand_m',    required=False,       type=int,     default=7,  help='magnitude in randaugment')
    argument_parser.add_argument('-rn', '--rand_n',         required=False,       type=int,      default=1,  help='# layers in randaugment')
    argument_parser.add_argument('-rat', '--ra_type',         required=False,       type=str,      default='Default',  help='# type of augmentation randaugment')
    argument_parser.add_argument('-val1', '--v1_type',         required=False,       type=str,      default='None',  help='# val set 1')
    argument_parser.add_argument('-val2', '--v2_type',         required=False,       type=str,      default='None',  help='# val set 2')
    argument_parser.add_argument('-test1', '--t1_type',         required=False,       type=str,      default='None',  help='# test set 1')
    argument_parser.add_argument('-test2', '--t2_type',         required=False,       type=str,      default='None',  help='# test set 2')
  
  
    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_dataset_dir= arguments['dataset_dir']
    parsed_output_dir = arguments['output_dir']
    parsed_experiment_tag = arguments['experiment_tag']
    parsed_augmentation_tag = arguments['augmentation_tag']
    parsed_organ_tag = arguments['organ_tag']
    parsed_network_tag = arguments['network_tag']
    parsed_trial_tag = arguments['trial_tag']
    parsed_n_epochs = arguments['n_epochs']
    parsed_batch_size = arguments['batch_size']
    parsed_lr = arguments['lr']
    parsed_workers = arguments['workers']
    parsed_test_only = eval(arguments['test_only'])
    parsed_ignore_std_test_set = eval(arguments['ignore_std_test_set'])
    parsed_prob_white_patch = eval(arguments['prob_white_patch'])
    parsed_apply_randaugment = eval(arguments['randaugment'])
    parsed_rand_m = arguments['rand_m']
    parsed_rand_n = arguments['rand_n']
    parsed_ra_type = arguments['ra_type']
    parsed_v1_type = arguments['v1_type']
    parsed_v2_type = arguments['v2_type']
    parsed_t1_type = arguments['t1_type']
    parsed_t2_type = arguments['t2_type']
    

    # Print parameters.
    #
    print(argument_parser.description)
    print('Dataset directory: {parsed_dataset_dir}'.format(parsed_dataset_dir=parsed_dataset_dir))
    print('Output directory: {parsed_output_dir}'.format(parsed_output_dir=parsed_output_dir))
    print('Experiment tag: {parsed_experiment_tag}'.format(parsed_experiment_tag=parsed_experiment_tag))
    print('Augmentation tag: {parsed_augmentation_tag}'.format(parsed_augmentation_tag=parsed_augmentation_tag))
    print('Organ tag: {parsed_organ_tag}'.format(parsed_organ_tag=parsed_organ_tag))
    print('Network tag: {parsed_network_tag}'.format(parsed_network_tag=parsed_network_tag))
    print('Trial tag: {parsed_trial_tag}'.format(parsed_trial_tag=parsed_trial_tag))
    print('Number of epochs: {parsed_n_epochs}'.format(parsed_n_epochs=parsed_n_epochs))
    print('Batch size: {parsed_batch_size}'.format(parsed_batch_size=parsed_batch_size))
    print('Learning rate: {parsed_lr}'.format(parsed_lr=parsed_lr))
    print('Number of workers: {parsed_workers}'.format(parsed_workers=parsed_workers))
    print('Test only: {parsed_test_only}'.format(parsed_test_only=parsed_test_only))
    print('Ignore standardization during testing: {parsed_ignore_std_test_set}'.format(parsed_ignore_std_test_set=parsed_ignore_std_test_set))
    print('Probability of white patch: {parsed_prob_white_patch}'.format(parsed_prob_white_patch=parsed_prob_white_patch))
    print('Apply randaugment: {parsed_apply_randaugment}'.format(parsed_apply_randaugment=parsed_apply_randaugment))
    print('Randaugment m parameter: {parsed_rand_m}'.format(parsed_rand_m=parsed_rand_m))
    print('Randaugment n parameter: {parsed_rand_n}'.format(parsed_rand_n=parsed_rand_n))
    print('Randaugment ra_type parameter: {parsed_ra_type}'.format(parsed_ra_type=parsed_ra_type))
    print('Val set 1: {parsed_v1_type}'.format(parsed_v1_type=parsed_v1_type))
    print('Val set 2: {parsed_v2_type}'.format(parsed_v2_type=parsed_v2_type))
    print('Test set 1: {parsed_t1_type}'.format(parsed_t1_type=parsed_t1_type))
    print('Test set 2: {parsed_t2_type}'.format(parsed_t2_type=parsed_t2_type))
    
    
    return parsed_dataset_dir, parsed_output_dir, parsed_experiment_tag, parsed_augmentation_tag, parsed_organ_tag, \
           parsed_network_tag, parsed_trial_tag, parsed_n_epochs, parsed_batch_size, parsed_lr, parsed_workers, \
           parsed_test_only, parsed_ignore_std_test_set, parsed_prob_white_patch, parsed_apply_randaugment, parsed_rand_m, \
           parsed_rand_n, parsed_ra_type, parsed_v1_type,parsed_v2_type, parsed_t1_type,parsed_t2_type

# ----------------------------------------------------------------------------------------------------

if __name__ == '__main__':




    # Collect arguments
    #
    print("In main")
    parsed_dataset_dir, parsed_output_dir, parsed_experiment_tag, parsed_augmentation_tag, parsed_organ_tag, \
    parsed_network_tag, parsed_trial_tag, parsed_n_epochs, parsed_batch_size, parsed_lr, parsed_workers, \
    parsed_test_only, parsed_ignore_std_test_set, parsed_prob_white_patch, parsed_apply_randaugment, parsed_rand_m, \
    parsed_rand_n, parsed_ra_type, parsed_v1_type,parsed_v2_type, parsed_t1_type,parsed_t2_type = collect_arguments()

    # Launch experiment
    #try:
    launch_experiment(
        dataset_dir=parsed_dataset_dir,
        output_dir=parsed_output_dir,
        experiment_tag=parsed_experiment_tag,
        augmentation_tag=parsed_augmentation_tag,
        organ_tag=parsed_organ_tag,
        network_tag=parsed_network_tag,
        trial_tag=parsed_trial_tag,
        workers=parsed_workers,
        batch_size=parsed_batch_size,
        lr=parsed_lr,
        n_epochs=parsed_n_epochs,
        test_only=parsed_test_only,
        ignore_std_test_set=parsed_ignore_std_test_set,
        prob_white_patch=parsed_prob_white_patch,
        randaugment=parsed_apply_randaugment,
        rand_m=parsed_rand_m,
        rand_n=parsed_rand_n,
        ra_type=parsed_ra_type, 
        v1_type=parsed_v1_type,
        v2_type=parsed_v2_type, 
        t1_type=parsed_t1_type,
        t2_type=parsed_t2_type
    )
    
    #except Exception as e:
    #print('Launch experiment failed with exception: {e}'.format(e=e), flush=True)



    try:
        time.sleep(random.randint(5, 30))
        exit_code = int(os.environ['CUDA_VISIBLE_DEVICES'])
        #print('CUDA_VISIBLE_DEVICES set exit code to {c}'.format(c=exit_code), flush=True)
    except:
        exit_code = 0

    exit(exit_code)
    
