import numpy as np
import os
import pickle
import logging
from optocoder.common.experiment import Experiment
from optocoder.basecalling.basecaller import Basecaller
from optocoder.data_management.image_management import ImageManager
from optocoder.evaluation.evaluate_basecalls import output_prediction_data, save_phasing_plot
from optocoder.machine_learning.ml_basecaller import ML_Basecaller

def run(parameters, rerun=False, only_report=False, run_ml=False, test_phasing=False, is_solid=False):
    """Main function to run Optocoder for the given parameters and run modes
    TODO: this section can be implemented nicer in the future

    Args:
        parameters (dict): Parameters of the optocoder run
        rerun (bool, optional): Run everything for the optical sequencing again. Defaults to False.
        only_report (bool, optional): Only report the plots etc.. Defaults to False.
        run_ml (bool, optional): Run machine learning pipeline. Defaults to False.
        test_phasing (bool, optional): Test phasing parameters during basecalling. Defaults to False.
        is_solid (bool, optional): Flag to set solid chemistry. This affects the matching etc. Defaults to False.
    """

    puck_id = parameters['puck_id'] # name of the puck
    num_cycles = parameters['num_cycles'] # number of optical seqeuncing cycles
    output_path = parameters['output_path'] # the location to save the results
    
    logging.info(f'Codebear is starting to run for puck {puck_id}')
    logging.info('--------------------------------------')
    logging.info(f'Number of cycles: {num_cycles} \n Output folder: {output_path}')
    
    # Create an output folder if it doesn't already exists
    if os.path.exists(output_path) == False:
        logging.warn('Output folder does not exist. Creating the folder...')
        os.makedirs(output_path)
        
    # Check if an experiment file exists
    experiment_exists = os.path.exists(os.path.join(output_path, 'experiment.pkl'))

    # Just load the experiment file and report the outcome
    # TODO: maybe we can have specific calls for basecalling etc.
    if only_report:
        if experiment_exists:
            with open(os.path.join(output_path, 'experiment.pkl'), 'rb') as input:
                logging.info('Only reporting the results by loading the experiment file...')
                experiment = pickle.load(input)
                experiment.report(output_path)
            logging.info(f"Experiment loaded and reported to {output_path}.")
            exit(0)
        else:
            logging.error('Experiment file does not exist!')
            exit(1)
    else:
        # Running optical sequencing from scratch
        if rerun == True or experiment_exists == False:
            experiment = optically_sequence(num_cycles, output_path, parameters, puck_id)

        # If we have an experiment file already and rerun is not set
        elif rerun == False and experiment_exists == True:
            with open(os.path.join(output_path, 'experiment.pkl'), 'rb') as input:
                experiment = pickle.load(input)

        # we can test phasing parameters to find the best values
        if test_phasing:
            logging.info('Testing phasing...')
            # create a basecaller and read the barcodes
            lig_seq = parameters['lig_seq'] if is_solid else None
            nuc_seq = parameters['nuc_seq'] if is_solid else None
            basecaller = Basecaller(num_cycles=num_cycles, nuc_order=parameters['nuc_order'], is_solid=is_solid, lig_seq=lig_seq, nuc_seq=nuc_seq)
            illumina_barcodes = np.genfromtxt(parameters['illumina_path'], dtype='str')
            # run phasing search
            parameter_results = basecaller.phasing_search(experiment.beads, illumina_barcodes)
            phasing_prob, prephasing_prob = next(k for k, v in parameter_results.items() if v == max(parameter_results.values(), key=lambda x: x))
            logging.info(f'Best phasing_probability: {phasing_prob}, best prephasing probability: {prephasing_prob}')
            phasing_save_path = os.path.join(output_path, 'phasing_grid.npy') # save the phasing match grid
            np.save(phasing_save_path, parameter_results) 
            logging.info('Phasing matching scores and heatmap are saved.')
            logging.info('Rerunning basecalling with the best phasing params.')
            basecaller = Basecaller(num_cycles=num_cycles, phasing=phasing_prob, prephasing=prephasing_prob, nuc_order=parameters['nuc_order'], is_solid=is_solid, lig_seq=lig_seq, nuc_seq=nuc_seq)
            basecaller.base_call(experiment.beads)
            logging.info('Saving the experiment and results.')
            experiment.save_experiment(output_path)
            experiment.report(output_path)
            logging.info('Done.')
            
        # Run machine learning basecaller
        if run_ml:
            lig_seq = parameters['lig_seq'] if is_solid else None
            nuc_seq = parameters['nuc_seq'] if is_solid else None
            run_ml_basecalling(parameters['illumina_path'], output_path, is_solid, lig_seq, nuc_seq)

    return 1

def optically_sequence(num_cycles, output_orig, parameters, puck_id, is_solid=False):
    """Optical sequencing run.
        Main steps are:
            1. Image registration and background correction
            2. Bead detection
            3. Crosstalk and phasing correction
            4. Basecalling
            5. Saving the results and plots
    Args:
        num_cycles (int): Number of optical sequencing cycles
        output_orig (str): Path to save the files
        parameters (dict): Parameters of the run
        puck_id (str): Name of the puck
        is_solid (bool, optional): Set if we are using solid chemistry data, e.g slideseq v1. Defaults to False.

    Returns:
        Experiment: an Experiment object with the results
    """
    logging.info('Optical sequencing run is starting...')

    image_manager = ImageManager(parameters['image_folder_path'], parameters['num_cycles'], channel_list=parameters['channel_list'], path_cycle_loc=parameters['file_name_seperator_loc'], output_path = output_orig)
    experiment = Experiment(image_manager=image_manager, puck_id=puck_id, num_cycles=num_cycles)

    logging.info('Detecting beads...')
    experiment.detect_beads()

    logging.info(f'Bead detection is completed. Number of detected beads: {len(experiment.beads)}')

    logging.info('Calculation bead intensities...')
    experiment.calculate_bead_intensities()
    
    logging.info('Basecalling started...')
    basecaller = Basecaller(num_cycles=num_cycles, nuc_order=parameters['nuc_order'], is_solid=is_solid)
    experiment.base_call(basecaller)
    logging.info('Basecalling is done.')
    logging.info('Saving the experiment and results.')
    experiment.save_experiment(output_orig)
    experiment.report(output_orig)
    logging.info('Optical sequencing run is done!')

    return experiment

def run_ml_basecalling(illumina_path, output_orig, is_solid, lig_seq=None, nuc_seq=None):
    """Run machine learning basecaller

    Args:
        illumina_path (str): Path of the illumina barcodes
        output_orig (str): Path of the saved results
        is_solid (bool): Set True if solid chemistry is used.
    """
    # load the experiment file
    with open(os.path.join(output_orig, 'experiment.pkl'), 'rb') as input:
        experiment = pickle.load(input)

    # read illumina barcodes
    illumina_barcodes = np.genfromtxt(illumina_path, dtype=str)

    # create the basecaller
    ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, output_orig, is_solid, lig_seq, nuc_seq)
    
    ml_basecaller.train('rnn')
    ml_basecaller.train('gb')
    ml_basecaller.train('mlp')
    ml_basecaller.train('rf')
    
    ml_basecaller.predict('gb')
    ml_basecaller.predict('mlp')
    ml_basecaller.predict('rnn')
    ml_basecaller.predict('rf')

    output_prediction_data(experiment.beads, experiment.num_cycles, output_orig, 'gb')
    output_prediction_data(experiment.beads, experiment.num_cycles, output_orig, 'mlp')
    output_prediction_data(experiment.beads, experiment.num_cycles, output_orig, 'rnn')
    output_prediction_data(experiment.beads, experiment.num_cycles, output_orig, 'rf')