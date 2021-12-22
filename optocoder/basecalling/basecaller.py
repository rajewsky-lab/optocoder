import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import quantile_transform, normalize, minmax_scale, scale, power_transform, robust_scale
import copy
import optocoder.basecalling.crosstalk_correction as cc
import optocoder.basecalling.phasing_correction as pc
import pandas as pd
from scipy.special import softmax
import logging
from joblib import Parallel, delayed
from optocoder.basecalling.utils import map_back, convert_barcodes, convert_to_color_space

class Basecaller:

    def __init__(self, num_cycles=12, phasing=0.07, prephasing=0.0, nuc_order=['C', 'A', 'T', 'G'], is_solid=False, lig_seq = None, nuc_seq=None):
        """This module processes detected beads and their intensities to generate the barcodes
        Currently there are three basecalling versions:
            - Naive: here we use just the highest intensity channel as the chosen base
            - Crosstalk correction: we first apply spectral crosstalk correction and then basecall
            - Phasing correction: we correct for phasing effects on top of the crosstalk correction
        
        Here, also we can test for phasing parameters to find the probability with most matches. This requires the
        existence of illumina barcodes.

        NOTE: in-house and slideseqv2 data can work directly. With solid chemistry data (slideseq v1), here we output
        as if what is called are barcodes but externally we match it to illumina in the colorspace. For details,
        see the paper or the slideseq v1 details.
        TODO: maybe this can also be done internatlly here.

        Args:
            num_cycles (int, optional): number of sequencing cycles. Defaults to 12.
            phasing (float, optional): amount of expected phasing. Defaults to 0.07.
            prephasing (float, optional): amount of expected prephasing. Defaults to 0.0.
            nuc_order (list, optional): order of the nucleotides. Defaults to ['C', 'A', 'T', 'G'].
            is_solid (bool, optional): if solid chemistry is used or not. Defaults to False.
            lig_seq (list, optional): if solid is used, we need to check the correct ligation sequence for matching. Defaults to None.
            nuc_seq (list, optional): if solid is used, we need to correct subset of cycles to eliminate constant sequences. Defaults to None.
        """
    
        self.num_cycles = num_cycles
        self.methods = ['naive', 'only_ct', 'phasing']
        self.default_phasing = phasing
        self.default_prephasing = prephasing
        self.crosstalk_matrix = None
        self.nuc_order = nuc_order
        self.is_solid = is_solid 
        self.lig_seq = lig_seq
        self.nuc_seq = nuc_seq

    def base_call(self, beads):
        """Main method to basecall with all the methods

        Args:
            beads (list): list of beads
        """
        # calculate the crosstalk matrix
        self.crosstalk_matrix = cc.calculate_crosstalk_matrix(beads)

        # calculate intensities the naive method
        logging.info("Basecalling with the naive method")
        naive_intensities = self._calculate_intensities(beads, correction_type='naive', crosstalk=False, phasing_prob=0.0, prephasing_prob=0.0)
        self._scale_and_set_intensities(beads, naive_intensities, correction_type='naive')
        self._assign_bead_barcodes(beads, 'naive_scaled', 'naive')
        logging.info('Naive basecalling is completed.')

        # calculate intensities for only crosstalk corrected basecalling
        logging.info('Basecalling with crosstalk correction')
        crosstalk_corrected_intensities = self._calculate_intensities(beads, correction_type='only_ct', crosstalk=True, phasing_prob=0.0, prephasing_prob=0.0)
        self._scale_and_set_intensities(beads, crosstalk_corrected_intensities, correction_type='only_ct')
        self._assign_bead_barcodes(beads, 'only_ct_scaled', 'only_ct')
        logging.info('Basecalling with crosstalk correction is completed.')

        # calculate intensities for phasing
        logging.info(f'Basecalling with phasing correction. Phasing prob: {self.default_phasing}, prephasing prob: {self.default_prephasing}')
        phasing_corrected_intensities = self._calculate_intensities(beads, correction_type='phasing', crosstalk=True, phasing_prob=self.default_phasing, prephasing_prob=self.default_prephasing)
        self._scale_and_set_intensities(beads, phasing_corrected_intensities, correction_type='phasing')
        self._assign_bead_barcodes(beads, 'phasing_scaled', 'phasing')
        logging.info('Basecalling with phasing correction is completed.')

    def _calculate_intensities(self, beads, correction_type, crosstalk=True, phasing_prob=0.07, prephasing_prob=0.0):
        """Calculate intensities of beads for a given correction method

        Args:
            beads (list): list of beads
            correction_type (str): type of correction
            crosstalk (bool, optional): if we use crosstalk correction or not. Defaults to True.
            phasing_prob (float, optional): expected phasing probability. Defaults to 0.07.
            prephasing_prob (float, optional): expected prephasing probability. Defaults to 0.0.

        Returns:
            ndarray: corrected intensities
        """
        crosstalk_matrix = self.crosstalk_matrix if crosstalk else np.identity(4) # create a crosstalk matrix if we are correcting for it
        phasing_matrix = pc.create_phasing_matrix(phasing_prob, prephasing_prob, self.num_cycles) # generate a phasing matrix

        # correct intensity values with given parameters
        corrected_intensities = self._correct_data(beads, crosstalk_matrix, phasing_matrix, correction_type=correction_type)
        
        return corrected_intensities

    def _correct_data(self, beads, cm, pm, correction_type='phasing'):
        """Apply corrections to the bead intensities

        Args:
            beads (list): list of beads
            cm (ndarray): crosstalk matrix
            pm (ndarray): phasing matrix
            correction_type (str, optional): type of the correction. Defaults to 'phasing'.

        Returns:
            ndarray: corrected bead intensities
        """
        # gather the intensities together
        # TODO: a bit ugly, requires cleaning
        final = []
        for cycle in range(self.num_cycles):
            n = cycle*4
            cycle_intensities = [np.asarray(bead.intensities['bc']).flatten()[[0+n,1+n,2+n,3+n]] for bead in beads]
            final.append(pd.DataFrame(cycle_intensities))
        final = pd.concat(final, axis=1)
        intensities = np.asarray(final)

        #inverse of the kronecker product of the phasing and crosstalk matrices
        A_init = np.kron(pm, cm)
        A_inv = np.linalg.inv(A_init)

        #correct the intensity values
        corrected_intensities = []
        for bead in intensities:
            # if it is the naive basecalling, there are no corrections
            if correction_type == 'naive':
                corrected_intensities.append(bead)
            # if we are doing the crosstalk or phasing correction, we need to multiply
            else:
                ints = np.matmul(A_inv, bead).flatten()
                corrected_intensities.append(ints)

        corrected_intensities = np.asarray(corrected_intensities)

        return corrected_intensities

    def _scale_and_set_intensities(self, beads, intensities, correction_type):
        """Apply scaling and set the bead intensities

        Args:
            beads (list): list of beads
            intensities (ndarray): corrected intensities
            correction_type (str): type of the correction to save the values
        """

        # apply robust scale to the data
        scaled_data = robust_scale(intensities)

        for i, bead in enumerate(beads):
            values = np.reshape(scaled_data[i,:], (self.num_cycles, 4))
            unscaled_values = np.reshape(intensities[i,:], (self.num_cycles, 4))
            for cycle in range(self.num_cycles):
                bead.set_intensities(unscaled_values[cycle], cycle, intensity_type=correction_type)
                bead.set_intensities(values[cycle], cycle, intensity_type=correction_type + '_scaled')

    def _assign_bead_barcodes(self, beads, intensity_type, method):
        """Assigns the barcodes to the beads

        Args:
            beads (list): list of beads
            intensity_type (str): type of intensity to use for the basecalling
            method (str): name of the basecalling method
        """
        for bead in beads:
            bead_barcode = []
            bead_scores = []

            for cycle_intensities in bead.intensities[intensity_type]:
                called_nucleotide, sims = self._base_call(softmax(cycle_intensities)) # apply softmax before calling
                bead_barcode.append(self.get_nuc_names(called_nucleotide))
                bead_scores.append(sims)

            bead.barcode[method] = ''.join(bead_barcode)
            bead.scores[method] = bead_scores
        
    def _base_call(self, intensities):
        """Base call and calculate the chastity score

        Args:
            intensities (array): intensities to basecall 

        Returns:
            int: index of the called channel
            list: scores of the basecalls
        """

        max_intensity_ch = np.argmax(intensities)
        scores = [0,0,0,0]
        s = sorted(intensities)
        chastity = s[-1] / (s[-1] + s[-2])
        chastity = np.clip(chastity, 0, 1)
        scores[max_intensity_ch] = chastity
        return max_intensity_ch, scores

    def phasing_search(self, beads, illumina_barcodes):
        """Phasing grid search to find the best phasing and prephasing parameters
        TODO: search space can be user defined
        Args:
            beads (list): list of beads
            illumina_barcodes (array): illumina barcodes

        Returns:
            dict: match counts for various phasing and prephasing params
        """

        self.crosstalk_matrix = cc.calculate_crosstalk_matrix(beads)
        logging.info('Starting phasing parameter search.')

        param_matches = dict()
        for phasing_prob in np.arange(0.0, 0.11, 0.01):
            for prephasing_prob in np.arange(0.0, 0.11, 0.01):
                param_matches[(phasing_prob, prephasing_prob)] = self.calculate_phasing_matches(beads, illumina_barcodes, phasing_prob, prephasing_prob)
        
        return param_matches

    def calculate_phasing_matches(self, beads, illumina_barcodes, phasing_prob, prephasing_prob):
        """Calculate phasing matches
        TODO: somethings are repetitive with other functions, a clean will be nice

        Args:
            beads (list): list of beads
            illumina_barcodes (array): list of illumina barcodes
            phasing_prob (float): phasing probability to test
            prephasing_prob (float): prephasing probability to test
        Returns:
            int: number of unique matches
        """

        phasing_corrected_intensities = self._calculate_intensities(beads, correction_type='phasing', crosstalk=True, phasing_prob=phasing_prob, prephasing_prob=prephasing_prob)

        scaled_data = robust_scale(phasing_corrected_intensities)
        scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], self.num_cycles, 4))
    
        barcodes = Parallel(n_jobs=32)(delayed(self.calculate_barcodes)(b) for b in scaled_data)	

        # if we have a solid chemistry sample, we need to do colorspace conversions
        if self.is_solid:
            # convert optocoder barcodes to colorspace
            colorspace_optical = [map_back(barcode) for barcode in barcodes]
            ligation_seq = self.lig_seq

            #convert illumina to colorspace
            illumina = []
            for barcode in illumina_barcodes:
                color_barcode = convert_barcodes(barcode, ligation_seq)
                illumina.append(''.join(color_barcode))

            # filter optical for the used cycles
            filtered_barcodes = []
            for barcode in colorspace_optical:
                nucs = np.array(list(barcode))
                nucs = nucs[self.nuc_seq]
                filtered_barcodes.append(''.join(nucs))
            barcodes = filtered_barcodes
            illumina_barcodes = illumina

        # get the number of matches
        num_matches = len(set(barcodes).intersection(illumina_barcodes))

        return num_matches

    def calculate_barcodes(self, bead):
        """Calculate barcodes
        TODO: repetitive, merge with other in the future

        Args:
            bead (Bead): a bead object

        Returns:
            str: bead barcode
        """
        bead_barcode = []

        for cycle_intensities in bead:
            called_nucleotide, sims = self._base_call(softmax(cycle_intensities))
            bead_barcode.append(self.get_nuc_names(called_nucleotide))

        return ''.join(bead_barcode)

    def get_nuc_names(self, idx):
        """Helper for getting the nuc letter
        Args:
            idx (int): index of the basecall

        Returns:
            str: nucleotide
        """
        if idx == -1:
            return 'N'
        letters = self.nuc_order
    
        return letters[idx]