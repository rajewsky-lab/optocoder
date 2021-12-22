from cv2 import imreadmulti
import os
import numpy as np
import cv2
from optocoder.image_analysis.utils import cdf, hist_matching
import optocoder.image_analysis.registration as registration
from skimage import data, restoration, util
import logging
import tifffile as tf
from optocoder.evaluation.evaluate_images import save_image
from skimage.metrics import structural_similarity as ssim

class ImageManager:
    def __init__(self, image_folder_path, num_cycles, channel_list=[0,1,4,5], path_cycle_loc=1, output_path=None):
        """ImageManager implements a few features to process microscopy images.
            - Reading the tif files from the folder in the correct sequence order.
            - Register an image to the given reference cycle.
            - Correct the background using morphological operations.

        Args:
            image_folder_path (str): Path for the image files
            num_cycles (int): Number of optical sequencing cycles
            channel_list (list, optional): Channels to use. In our in-house method, we have
                6 channels but 2 of them are just internal controls. Defaults to [0,1,4,5].
            path_cycle_loc (int, optional): The location of the underscore where the cycle id is specific.
                This is just to sort the image names to be sure that cycles are in the correct order. Defaults to 1.
            output_path (str, optional): Path to save things.
        """
        self.image_folder_path = image_folder_path # path of the images
        self.num_cycles = num_cycles # number of sequencing cycles
        self.channel_list = channel_list # list of image channels to be used
        self.path_cycle_loc = path_cycle_loc # location of the cycle identifier in the file name
        self.output_path = output_path

        self.image_paths = self._read_image_paths(image_folder_path, path_cycle_loc) # read image paths in the cycle order
        self.warp_params = [] # warping parameters of every cycle
        self.reg_similarity_to_ref = [] # similarity scores after registration
        self.unreg_similarity_to_ref = [] # similarity scores before registration

    def _read_image_paths(self, folder_path, cycle_loc):
        """Gets tif image paths from the folder and sorts them

        Args:
            folder_path (str): Path of the image folder
            cycle_loc (int): Location of the cycle order seperator

        Returns:
            list: Sorted image paths
        """

        image_paths = []
        image_names = []
        for file in os.listdir(folder_path):
            if file.endswith(".tif"):
                image_names.append(file)

        #Sort the images in the folder
        image_names = sorted(image_names, key=lambda x:int(self.only_numerics(x.split('_')[cycle_loc])))
        for image in image_names[:self.num_cycles]:
            image_paths.append(os.path.join(folder_path, image))
        
        logging.info('Image paths are loaded. Please check if the cycle order is correct!')
        logging.info(image_names)
        return image_paths

    def _read_image(self, cycle_id):
        """Read the tif image for a given cycle

        Args:
            cycle_id (image): cycle to read

        Returns:
            list: list of channel images 
            ndarray: overlay image
        """
        try:
            image = np.asarray(tf.imread(self.image_paths[cycle_id]))
        except FileNotFoundError:
            print("Image file is missing!")
            exit(1)
        
        # create an overlap image
        overlay_image1 = cv2.add(image[self.channel_list[0]], image[self.channel_list[1]])
        overlay_image2 = cv2.add(image[self.channel_list[2]], image[self.channel_list[3]])
        overlay_image = cv2.add(overlay_image1, overlay_image2)

        # get the channel images for only the necessary ones
        image = image[self.channel_list, ...]

        return image, self._convert_8bit(overlay_image)

    def register_image_to_cycle(self, cycle_id, reference_cycle_id, hist_matching_on=True):
        """Register a cycle image to the reference cycle

        Args:
            cycle_id (int): cycle to register
            reference_cycle_id (int): reference cycle
            hist_matching_on (bool, optional): Histogram matching or not. Defaults to True.

        Returns:
            ndarray: registered overlay image
            list: registered channel images
        """
        # Read reference cycle image. we just need the overlay for the registration
        _, reference_cycle_overlay = self._read_image(reference_cycle_id)

        reference_cdf = cdf(reference_cycle_overlay) # calculate the cdf for hist eq

        cycle_channels, cycle_overlay = self._read_image(cycle_id)
        save_image(cycle_overlay, cycle_id, self.output_path, 'raw_overlays')

        if hist_matching_on:
            matched_image = hist_matching(cdf(cycle_overlay), reference_cdf, cycle_overlay) 
            save_image(matched_image, cycle_id, self.output_path, 'hist_eq_overlays')
            registered_overlay, registered_nucleotides, warp = registration.register_image(reference_cycle_overlay, matched_image, cycle_channels)
        else:
            registered_overlay, registered_nucleotides, warp = registration.register_image(reference_cycle_overlay, cycle_overlay, cycle_channels)

        save_image(registered_overlay, cycle_id, self.output_path, 'raw_registered_overlays')
        self.warp_params.append(warp) # keep the warping parameters for later use

        # calculate similarity scores
        sim_score_unreg = ssim(reference_cycle_overlay, cycle_overlay, data_range=cycle_overlay.max() - cycle_overlay.min())
        sim_score_reg = ssim(reference_cycle_overlay, registered_overlay, data_range=registered_overlay.max() - registered_overlay.min())
        self.reg_similarity_to_ref.append(sim_score_reg)
        self.unreg_similarity_to_ref.append(sim_score_unreg)
        return registered_overlay, registered_nucleotides

    def correct_background(self, channels):
        """Correct background using morphological operations. The basic idea is to 
        detect the foreground and background with a big kernel.

        Args:
            channels (list): channel images

        Returns:
            list: channel images after background correction
        """
        
        #Get a super big kernel. Ideally circular would be the best but rect performs similarly
        #and it is much faster
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(64,64))

        #Iterate through the cycle images
        corrected_images = []
        for img in channels:
            #Get the background by applying opening operation
            b = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            corrected_image = cv2.subtract(img,b) 
            corrected_images.append(corrected_image)
        return corrected_images

    def get_overlay(self, cycle):
        """Get the overlay image

        Args:
            cycle (int): cycle to get overlay for

        Returns:
            ndarray: overlay image
        """
        _, overlay = self._read_image(cycle)

        return overlay

    def _convert_8bit(self, image):
        """Convert image to 8bit for later image processing

        Args:
            image (ndarray): image to convert

        Returns:
            ndarray: 8 bit normalized image
        """
        image = image - image.min()
        image = image / image.max() * 255
        return np.uint8(image)
        
    def only_numerics(self, seq):
        """This is a helper to get the digits from the file name"""
        seq_type = type(seq)
        return seq_type().join(filter(seq_type.isdigit, seq))