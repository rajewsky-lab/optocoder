import kerastuner
import numpy as np
from sklearn import model_selection

#https://github.com/keras-team/keras-tuner/issues/122#issuecomment-545090991
class CVTuner(kerastuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=32, epochs=1):
        cv = model_selection.KFold(5)
        val_losses = []

        for train_indices, test_indices in cv.split(x):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)