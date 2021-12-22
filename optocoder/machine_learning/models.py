from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from keras import backend as K
from scipy import stats
import kerastuner as kt

def get_model(model):
    if model == 'rf':
        parameters = {'estimator__bootstrap': [True, False],
                'estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
                'estimator__max_features': ['auto', 'sqrt'],
                'estimator__min_samples_leaf': [1, 2, 4],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__n_estimators': [130, 180, 230, 500, 1000]}
        model = RandomForestClassifier()
        return model, parameters

    elif model == 'mlp':
        parameters = {
        'estimator__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'estimator__activation': ['tanh', 'relu'],
        'estimator__solver': ['sgd', 'adam'],
        'estimator__alpha': [0.0001, 0.05, 0.1],
        'estimator__learning_rate': ['constant','adaptive'],
        'estimator__max_iter': [200,400,600,1000]
        }
        model = MLPClassifier()
        return model, parameters

    elif model == 'gb':
        parameters = {'estimator__n_estimators': stats.randint(150, 1000),
            'estimator__learning_rate': stats.uniform(0.01, 0.59),
            'estimator__subsample': stats.uniform(0.3, 0.6),
            'estimator__max_depth': [3, 4, 5, 6, 7, 8, 9],
            'estimator__colsample_bytree': stats.uniform(0.5, 0.4),
            'estimator__min_child_weight': [1, 2, 3, 4],
            'estimator__tree_method': ['hist']
            }
        model = XGBClassifier()
        return model, parameters

class RNNHyperModel(kt.HyperModel):

    def __init__(self, num_cycles):
        self.num_cycles = num_cycles

    def build(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                                return_sequences=True), input_shape=(self.num_cycles, 4)))
        model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))

        model.add(keras.layers.Dense(4, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
        
def get_rnn_model(num_cycles, hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                            return_sequences=True), input_shape=(num_cycles, 4)))
    model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))

    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model