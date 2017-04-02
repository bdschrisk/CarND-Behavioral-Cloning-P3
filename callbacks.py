from keras.callbacks import Callback, EarlyStopping
import numpy as np

class EarlyStoppingByDivergence(EarlyStopping):
    """ Defines an early stopping callback which terminates training if there is sign of overfitting """
    def __init__(self, train_monitor = 'loss', val_monitor = 'val_loss', ratio = 0.0, patience = 0, verbose = 0):
        super(EarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_monitor = train_monitor
        self.val_monitor = val_monitor
        self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get(self.val_monitor)
        current_train = logs.get(self.train_monitor)
        if current_val is None or current_train is None:
            warnings.warn('Early stopping requires %s available!' % (self.monitor), RuntimeWarning)
        
        # check for resuming or first epoch
        if (len(self.train_losses) > 0 and len(self.val_losses) > 0):
            # If training loss is decreasing while val loss is increasing
            # Loss (t-1) > Loss (t0), Loss_val (t0) > Loss_val (t0)
            if self.monitor_op(self.train_losses[-1], current_train) and self.monitor_op(current_val, self.val_losses[-1]):
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                self.wait += 1
            else:
                self.wait = 0
        
        self.train_losses.append(current_train)
        self.val_losses.append(current_val)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: Aborting training as model overfitting.' % (self.stopped_epoch))