import argparse
import yaml
from optocoder import run_optocoder
import logging
import os

# set logging parameters
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)  


def _parse_run_config_params():
    """Parse the paramaters defined from the console input of the user
    Returns:
        namespace: Namespace for the parsed console args
    """

    parser = argparse.ArgumentParser(description="Run Optocoder for an optical sequencing experiment")
    parser.add_argument('-config', type=str, help='Config file path')
    parser.add_argument('--rerun', action='store_true', help='Set to rerun optical sequencing pipeline')
    parser.add_argument('--only_report', action='store_true', help='Set to only save result files and plots')
    parser.add_argument('--run_ml', action='store_true', help='Set to run ml basecaller')
    parser.add_argument('--test_phasing', action='store_true', help='Set to test phasing parameters for correction')
    args = parser.parse_args()

    # check if the config file is there!
    if args.config is None:
        logging.error('Config file must be provided!')
        exit(1)

    # if only report is called, we can not do other things. TODO: this can be nicer in the future
    if args.only_report and (args.run_ml or args.test_phasing or args.rerun):
        logging.error('Only report can only be called alone.')
        exit(1)

    return args

def _parse_config_file(file_path):
    """Parse config yaml for the experiment details

    Args:
        file_path (str): The file path for the config yaml file
    Returns:
        dict: Run parameters
    """

    try:
        with open(file_path, 'r') as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("Config file does not exist!")
        exit(1)   

    parameters = dict()
    parameters['image_folder_path'] = config_file['image_folder_path']  # the path of the images
    parameters['file_name_seperator_loc'] = config_file['file_name_seperator_loc'] # seperator for the file names
    parameters['num_cycles'] = config_file['num_cycles'] # number of sequencing cycles
    parameters['puck_id'] = config_file['puck_id'] # id of the puck/experiment
    parameters['output_path'] = config_file['output_path'] # path to save the output
    parameters['channel_list'] = config_file['channels_to_use'] # channels to use (e.g when there are more channels than 4)
    parameters['nuc_order'] = config_file['nuc_order'] # order of the nucs to match in the barcodes

    # add if illumina path exists
    if 'illumina_path' in config_file:
        parameters['illumina_path'] = config_file['illumina_path']

    # if the solid chemistry is used such as slideseq
    if 'is_solid' in config_file:
        parameters['is_solid'] = config_file['is_solid']
        parameters['nuc_seq'] = config_file['nuc_seq']
        parameters['lig_seq'] = config_file['lig_seq']
    else: 
        parameters['is_solid'] = False

    return parameters 
    
if __name__ == '__main__':

    args = _parse_run_config_params() # we read the args from the console
    parameters = _parse_config_file(args.config) # read the experiment parameters from the yaml file

    # create output folder if it doesn't exist
    if os.path.exists(parameters['output_path']) == False:
        os.makedirs(parameters['output_path'])
    
    # set the logger output
    fh = logging.FileHandler(os.path.join(parameters['output_path'], 'experiment.log'))
    fh.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(fh)

    ### RUN OPTOCODER
    run_optocoder.run(parameters, args.rerun, args.only_report, args.run_ml, args.test_phasing, parameters['is_solid'])
