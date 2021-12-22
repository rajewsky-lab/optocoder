import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from optocoder.machine_learning.models import get_model, get_rnn_model, RNNHyperModel
from optocoder.machine_learning.CVTuner import CVTuner
import kerastuner
import logging
import pickle
import os
from scipy import stats
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import tensorflow as tf
from optocoder.basecalling.utils import map_back, convert_barcodes, convert_to_color_space
from functools import partial

def get_letter(barcode):
    """Helper function to convert predictions to barcodes"""
    d = ['A', 'C', 'G', 'T']
    bc = []
    for element in list(barcode):
        bc.append(d[int(element)])
    return ''.join(bc)

class ML_Basecaller():

    def __init__(self, experiment, illumina_barcodes, output_path, is_solid=False, lig_seq = None, nuc_seq=None):
        """Machine learning module to predict bases for non-matching barcodes

        Args:
            experiment (Experiment): the Experiment object to train the machine learning model
            illumina_barcodes (list): list of illumina barcodes
            output_path (str): path to save things
            is_solid (bool, optional): if we use the solid chemistry samples. Defaults to False.
            lig_seq (ndarray, optional): ligation sequence if we are using solid barcodes. Defaults to None.
            nuc_seq (ndarray, optional): nucleotide sequence if we are using solid barcodes. Defaults to None.
        """
        
        self.experiment = experiment # the experiment to test the ml models on
        self.illumina_barcodes = illumina_barcodes
        self.output_path = output_path
        self.is_solid = is_solid
        self.lig_seq = lig_seq
        self.nuc_seq = nuc_seq
        self.matching_beads, self.all_beads = self.get_matching_beads(illumina_barcodes, is_solid) # get the matching beads and all beads
        self.classifiers = {} # the dictionary to keep the best models

    def get_matching_beads(self, illumina_barcodes, is_solid):
        """Filter for the matching beads"""

        bead_ids = [bead.id for bead in self.experiment.beads]
        barcodes = [bead.barcode["phasing"] for bead in self.experiment.beads]

        intensities = [np.asarray(bead.intensities['naive_scaled'])[:,[0,1,2,3]].flatten() for bead in self.experiment.beads]

        d = {"bead_id": bead_ids, "barcodes": barcodes}
        df = pd.DataFrame(d)
        df_intensities = pd.DataFrame(intensities, columns=['cycle_%i_ch_%i' % (i, j) for i in
                                                            range(1, self.experiment.num_cycles+1) for j in
                                                            range(1, 5)])

        optical_data = pd.concat([df, df_intensities], axis=1)

        if is_solid:
            colorspace_optical = [map_back(barcode) for barcode in barcodes]
            ligation_seq = self.lig_seq
            illumina = []
            for barcode in illumina_barcodes:
                color_barcode = convert_barcodes(barcode, ligation_seq)
                illumina.append(''.join(color_barcode))

            filtered_barcodes = []
            for barcode in colorspace_optical:
                nucs = np.array(list(barcode))
                nucs = nucs[self.nuc_seq]
                filtered_barcodes.append(''.join(nucs))
            barcodes = filtered_barcodes
            illumina_barcodes = illumina
            matching_beads = optical_data[pd.Series(barcodes).isin(illumina_barcodes)]
        else:
            matching_beads = optical_data[optical_data['barcodes'].isin(illumina_barcodes)]

        return matching_beads, optical_data

    @staticmethod
    def _prepare_data(dataset, num_cycles):
        """Prepare the labeled dataset for training and prediction"""

        ys = dataset['barcodes']
        ys = np.asarray([list(y) for y in ys])
        ys = np.vstack(ys)
        ys = ys[..., np.newaxis]

        xs = dataset.iloc[:, 2:]
        xs = np.asarray(xs)
        xs = np.hsplit(xs, num_cycles)
        xs = np.dstack(xs)

        xs = np.transpose(xs, axes=[0, 2, 1])
 
        le = LabelEncoder()
        for i in range(num_cycles):
            ys[:, i] = le.fit_transform(ys[:, i]).reshape(ys[:, i].shape)

        ys = ys.astype(int)

        return xs, ys

    def _train_rnn(self, xs, ys, cv=False):
        # Train a RNN model with random parameter search
        rnn_model = RNNHyperModel(self.experiment.num_cycles)

        if cv:
            tuner = CVTuner(
                hypermodel=rnn_model,
                oracle=kerastuner.oracles.RandomSearch(
                objective='val_loss',
                max_trials=10), directory=os.path.join(self.output_path, 'ml', 'intermediate'), project_name=self.experiment.puck_id)
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=2)

            tuner.search(xs, ys,
                            epochs=30, batch_size=50)
        else:
            X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.20, shuffle=True)

            tuner = RandomSearch(
            rnn_model,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory=os.path.join(self.output_path, 'ml', 'intermediate'),
            project_name=self.experiment.puck_id)
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=2)
            tuner.search(X_train, y_train,
                     epochs=30, batch_size=50,
                     validation_data=(X_test, y_test), callbacks=[stop_early])


        models = tuner.get_best_models(num_models=1)
        return models[0]

    def _train(self, xs, ys, model, cv=False):
        # sklearn classifier requires a flattened feature array
        xs_flattened = xs.reshape((xs.shape[0],xs.shape[1]*xs.shape[2]))
        ys_flattened = np.squeeze(ys)

        logging.info(f'Training a {model}.')
        
        # Get the model and hyperparameter ranges
        model_func, model_param_range = get_model(model=model)
            
        # Create a multioutputclassifier that can train for all cycles at once
        clf = MultiOutputClassifier(model_func)

        # Random search of hyper parameters
        if cv:
            cv_val = 5
        else:
            cv_val = ShuffleSplit(n_splits=1, test_size=0.2)
        randm = RandomizedSearchCV(estimator=clf, param_distributions = model_param_range, cv = cv_val , n_iter = 10, verbose=1, n_jobs=16) 
        randm.fit(xs_flattened, ys_flattened)
        
        # Get the best_model
        best_model = randm.best_estimator_
            
        # Get cv accuracy results
        results = randm.cv_results_
        directory=os.path.join(self.output_path, 'ml', 'intermediate', f'{model}_cv_results.pickle')
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        with open(directory, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Get the best params
        best_params = randm.best_params_
        
        logging.info(f'{model} is trained.')

        return best_model

    def train(self, model_type):
        
        # create training and validation data
        dataset = self.matching_beads
        xs, ys = self._prepare_data(dataset, self.experiment.num_cycles)
        
        if model_type == 'rnn': 
            model = self._train_rnn(xs, ys)
        else:
            model = self._train(xs, ys, model_type)

        # Keep the results
        self.classifiers[model_type] = model

        if model_type != 'rnn':
            directory=os.path.join(self.output_path, 'ml', 'intermediate', f'{model_type}.pickle')
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            with open(directory, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'{model_type} model is saved to the ml folder.')

    def _predict_rnn(self, xs, ys):
        probabilites = self.classifiers['rnn'].predict_proba(xs)
        predictions = self.classifiers['rnn'].predict_classes(xs)
        return predictions, probabilites

    def _predict(self, xs, ys, model_type):
        xs_flattened = xs.reshape((xs.shape[0],xs.shape[1]*xs.shape[2]))
        ys_flattened = np.squeeze(ys)

        probabilites = self.classifiers[model_type].predict_proba(xs_flattened)
        predictions = self.classifiers[model_type].predict(xs_flattened)
        probabilites = np.array(probabilites).transpose(1, 0,2)
        return predictions, probabilites

    def predict(self, model_type):
        """Predict the barcodes for all beads"""

        non_matching_ids = self.all_beads[~self.all_beads['bead_id'].isin(self.matching_beads['bead_id'])]['bead_id']
        dataset = self.all_beads
        xs, ys = self._prepare_data(dataset, self.experiment.num_cycles)

        if model_type == 'rnn':
            predictions, probabilites = self._predict_rnn(xs, ys)
        else:
            predictions, probabilites = self._predict(xs, ys, model_type)

        barcodes = []
        for i, (bc_pred, bc_orig) in enumerate(zip(predictions, ys)):
            if i in non_matching_ids:
                bc_l = get_letter(bc_pred)
            else:
                bc_l = get_letter(bc_orig)
            barcodes.append(bc_l)
        
        for bead, barcode, prob in zip(self.experiment.beads, barcodes, probabilites):
            bead.barcode[model_type] = barcode
            bead.scores[model_type] = prob
        return barcodes, probabilites
  