"""
This module contains useful classes and functions to support Deep Learning and Keras operations.
"""

import matplotlib as mpl

from metrics import accuracy_fn

mpl.use('Agg')  # plot figures when no screen available


from keras import callbacks, layers, losses, metrics, activations, regularizers, optimizers, models, utils, applications
import keras.backend as K

import warnings
import pandas as pd
from os.path import basename, dirname, exists, join, splitext
from matplotlib import pyplot as plt
import os
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, accuracy_score


#----------------------------------------------------------------------------------------------------

def fit_model(train_gen, val_gen, model, n_epochs, output_dir, train_step_multiplier=1.0, workers=1,
              loss_list=['loss', 'val_loss'], metric_list=['binary_accuracy', 'val_binary_accuracy'],
              allow_resume_training=True, custom_objects=None, patience=4, extra_callbacks=[],
              monitor='val_loss', mode='auto', min_lr=1e-5, steps_per_epoch=None, warm_up_model=False):
    """
    Trains a model to fit the data from the data generator.

    Args:
        train_gen (generator): generator providing training batches.
        val_gen (Sequence or None): sequence providing validation batches.
        model (Keras model): model with neural network.
        n_epochs (int): number of epochs to run.
        workers (int): number of parallel workers to use.
        train_step_multiplier (float): multiply steps to take during 1 epoch of training.
        output_dir (str): output directory to store results (it will be created if needed).
        loss_list (list): loss names to plot.
        metric_list (list): metric names to plot.
        allow_resume_training (bool): True to resume training if saved model found in output folder.
        custom_objects (dict or None): custom objects for the Keras model loader (if resuming training).
        patience (int): wait epochs before reducing learning rate.
        extra_callbacks (list): extra callbacks to add.
        monitor (str): training parameter to monitor for callbacks (checkpoint, etc.).
        mode (str): 'max', 'min' or 'auto' to monitor the training parameter.
        min_lr (float): stops training if learning rate is reduced to this value.

    """

    # Prepare directory
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Continue training if model found
    epochs_run = 0
    if exists(join(output_dir, 'last_epoch.h5')) and exists(join(output_dir, 'history.csv')) and allow_resume_training:
        model = models.load_model(join(output_dir, 'last_epoch.h5'), custom_objects=custom_objects)
        df = pd.read_csv(join(output_dir, 'history.csv'))
        epochs_run = len(df)
        print('Resuming training from saved model ...', flush=True)

    else:

        # If new model and warm_up required, train for a few epochs, then unfreeze layers and resume training
        if warm_up_model:

            # Train model
            print('Warming up model...', flush=True)
            model.fit_generator(
                generator=train_gen,
                steps_per_epoch=200,
                epochs=1,
                verbose=2,  # TODO log errors only
                callbacks=None,
                validation_data=None,
                validation_steps=None,
                initial_epoch=0,
                max_queue_size=10,
                workers=workers,
                use_multiprocessing=True if workers > 1 else False
            )

            # Unfreeze
            print('Setting all layers to trainable...', flush=True)
            for layer in model.layers:
                layer.trainable = True

            # Recompile
            model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics
            )

        if epochs_run < n_epochs:

            print('Defining callbacks starts...')
            # Define callbacks
            callback_list = [
                StoreModelSummary(filepath=join(output_dir, 'model_summary.txt'), verbose=1),
                HistoryCsv(file_path=join(output_dir, 'history.csv'))
            ]
            callback_list.extend(extra_callbacks)
            callback_list.extend([
                ModelCheckpointProtected(
                    history_path=join(output_dir, 'history.csv'),
                    filepath=join(output_dir, 'checkpoint.h5'),
                    monitor=monitor,
                    mode=mode,
                    verbose=2,
                    save_best_only=True
                ),
                ModelCheckpointProtected(
                    history_path=join(output_dir, 'history.csv'),
                    filepath=join(output_dir, 'last_epoch.h5'),
                    monitor=monitor,
                    mode=mode,
                    verbose=1,
                    save_best_only=False
                ),
                ReduceLROnPlateauProtected(
                    history_path=join(output_dir, 'history.csv'),
                    monitor=monitor,
                    mode=mode,
                    factor=1.0/10,
                    patience=patience,
                    verbose=1,
                    cooldown=2,
                    min_lr=min_lr
                ),
                PlotHistory(
                    plot_path=join(output_dir, 'history.png'),
                    log_path=join(output_dir, 'history.csv'),
                    loss_list=loss_list,
                    metric_list=metric_list
                ),
                FindBestEval(
                    history_path=join(output_dir, 'history.csv'),
                    output_path=join(output_dir, 'history_best_row.csv'),
                    monitor=monitor,
                    mode=mode
                ),
                FinishedFlag(
                    file_path=join(output_dir, 'training_finished.txt')
                )
            ])

            # Train model
            print('Training starts...')
            model.fit_generator(
                generator=train_gen,
                steps_per_epoch=int(len(train_gen)*train_step_multiplier) if steps_per_epoch is None else np.min([steps_per_epoch, int(len(train_gen)*train_step_multiplier)]),
                epochs=n_epochs,
                verbose=2,  # TODO log errors only
                callbacks=callback_list,
                validation_data=val_gen,
                validation_steps=len(val_gen) if val_gen is not None else None,
                initial_epoch=epochs_run,
                max_queue_size=10,
                workers=workers,
                use_multiprocessing=True if workers > 1 else False
            )

            #except KilledException as e:

            # Notify
            #print('SIGINT detected while training model in {path}. Exiting.'.format(path=output_dir), flush=True)
            #open(join(output_dir, 'interrupted.lock'), 'a').close()
            #raise KilledException()

    return model

#----------------------------------------------------------------------------------------------------


def pred_model(data_seq, model_path, output_path, steps=None, workers=1):
    """
    Make predictions on a given dataset given a trained model.

    Args:
        data_seq (Sequence): sequence providing batches.
        model_path (str): path to saved model.
        output_path (str): path to store the numpy array with predictions.
        steps (int or None): batches to evaluate (use None for all available).

    Returns: array with predictions.

    """

    # Load model
    model = models.load_model(model_path)

    # Predict
    preds = model.predict_generator(
        generator=data_seq,
        steps=len(data_seq) if steps is None else steps,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,  # disabled due to OverflowError: cannot serialize a bytes object larger than 4 GiB
        verbose=0
    )

    # Save
    np.save(output_path, preds)
    return preds

#----------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------

def compute_accuracy(labels, preds, output_path):

    # df = pd.read_csv(input_path)
    # labels = np.array(df['labels'].values)
    # preds = np.array(df['preds'].values)
    acc = accuracy_fn(labels, preds)
    pd.DataFrame([{'accuracy': acc}]).to_csv(output_path)

#----------------------------------------------------------------------------------------------------

class StoreModelSummary(callbacks.Callback):
    """
    Keras callback to store the model summary to a text file.
    """

    def __init__(self, filepath, verbose=0):
        class CaptureLines(object):
            def __init__(self):
                self.lines = []

            def put_line(self, line):
                self.lines.append(line)

            def get_lines(self):
                return self.lines

        self.filepath = filepath
        self.verbose = verbose
        self.capture = CaptureLines()
        super(StoreModelSummary, self).__init__()

    def on_train_begin(self, logs=None):

        # Get summary
        self.model.summary(print_fn=self.capture.put_line)
        lines = '\n'.join(self.capture.get_lines())

        # Store
        with open(self.filepath, 'w') as f:
            f.write(lines)

        # Print
        if self.verbose:
            print(lines)

#----------------------------------------------------------------------------------------------------

class HistoryCsv(callbacks.Callback):
    """
    Keras callback that logs training values safely (flushing to disk always). It also saves the learning rate.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        super(HistoryCsv, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        logs_row = logs.copy()
        logs_row['epoch'] = epoch
        try:
            logs_row['lr'] = float(K.get_value(self.model.optimizer.lr))

        except Exception as e:
            print('Could not retrieve learning rate value.', flush=True)

        if exists(self.file_path):
            df = pd.read_csv(self.file_path)

            df = df.append(pd.Series(logs_row), ignore_index=True)
        else:
            df = pd.DataFrame(logs_row, index=[0])

        try:
            df.to_csv(self.file_path)

        except KilledException as e:

            df.to_csv(self.file_path)
            raise KilledException()

#----------------------------------------------------------------------------------------------------

class FinishedFlag(callbacks.Callback):
    """
    Keras callback writes a file to disk when training mode completed.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        super(FinishedFlag, self).__init__()

    def on_train_end(self, logs=None):
        open(join(self.file_path), 'a').close()

#----------------------------------------------------------------------------------------------------

def plot_history(history_path, loss_list, metric_list, plot_path):
    """
    Makes a plot of training history. Top is a logplot with loss values. The learning rate is also plotted in top using
    the right axis. Bottom is a plot with metrics.

    Args:
        history_path (str): path to CSV file containing the training history (see HistoryCsv).
        loss_list (list): list of losses to plot.
        metric_list (list): list of metrics to plot.
        plot_path (str): output path to store PNG with history plot.
    """

    # Format data
    df = pd.read_csv(history_path)

    # Subplots: losses and metrics
    fig, (ax_loss, ax_metric) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot learning rate in right axis
    if 'lr' in df.columns:
        ax_lr = ax_loss.twinx()
        lines = ax_lr.semilogy(df.index.values, df.loc[:, 'lr'], '-k', label='lr')
        ax_lr.set_ylabel('Learning rate')
    else:
        lines = None

    # Plot losses
    for loss in loss_list:
        try:
            new_lines = ax_loss.semilogy(df.index.values, df.loc[:, loss], label=loss)
            if lines:
                lines += new_lines
            else:
                lines = new_lines
        except:
            pass

    # Final touches
    ax_loss.legend()
    ax_loss.grid()
    ax_loss.set_title('Training Summary')
    ax_loss.set_ylabel('Loss')
    lines_labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, lines_labels, loc=0)

    # Plot metrics
    for metric in metric_list:
        try:
            ax_metric.plot(df.index.values, df.loc[:, metric], label=metric)
        except:
            pass

    # Final touches
    ax_metric.legend()
    ax_metric.grid()
    ax_metric.set_ylabel('Metric')
    # ax_metric.set_xlabel('Number of epochs')
    ax_metric.legend(loc=0)

    # Store plot in disk
    plt.savefig(plot_path)
    plt.close()

#----------------------------------------------------------------------------------------------------

class PlotHistory(callbacks.Callback):
    """
    Keras callback that plots training history.
    """

    def __init__(self, plot_path, log_path, loss_list, metric_list):
        self.plot_path = plot_path
        self.loss_list = loss_list
        self.metric_list = metric_list
        self.log_path = log_path
        super(PlotHistory, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        try:
            plot_history(self.log_path, self.loss_list, self.metric_list, self.plot_path)
        except Exception as e:
            print('Plotting failed. Exception: %s' % str(e))

#----------------------------------------------------------------------------------------------------

class FindBestEval(callbacks.Callback):
    """
    Keras callback that finds the history row with the best monitored metric.
    """

    def __init__(self, history_path, output_path, monitor, mode):
        self.history_path = history_path
        self.output_path = output_path
        self.monitor = monitor
        self.mode = mode

        if self.mode == 'min':
            self.mode_fn = np.argmin
        elif self.mode == 'max':
            self.mode_fn = np.argmax
        else:  # auto
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.mode_fn = np.argmax
            else:
                self.mode_fn = np.argmin

        super(FindBestEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Read history
            df = pd.read_csv(self.history_path)

            # Check entries
            if len(df.loc[df[self.monitor].notnull(), :]) > 0:

                # Best row
                best_row = df.loc[self.mode_fn(df.loc[df[self.monitor].notnull(), self.monitor]), :]

                # Save
                best_row.to_csv(self.output_path)

        except Exception as e:
            print('Storing best row in history failed. Exception: %s' % str(e))

#----------------------------------------------------------------------------------------------------

class ReduceLROnPlateauProtected(callbacks.Callback):
    """Reduce learning rate when a metric has stopped improving. Reads custom history file. Stops training
    if LR is min LR. """

    def __init__(self, history_path, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateauProtected, self).__init__()

        self.history_path = history_path
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.best_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
            self.best_op = lambda x: np.min(x)
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
            self.best_op = lambda x: np.max(x)
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        # current = logs.get(self.monitor)
        df = pd.read_csv(self.history_path)
        current = df.loc[len(df) - 1, self.monitor]
        self.best = self.best if len(df.loc[df[self.monitor].notnull(), :]) <= 1 else self.best_op(df.loc[df[self.monitor].notnull(), self.monitor].values[:-1])  # checks best score from history (useful for resuming) excluding current result
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr + self.lr_epsilon:

                        new_lr = old_lr * self.factor

                        # Stop training if LR is min
                        if new_lr <= self.min_lr:
                            if self.verbose > 0:
                                print('\nEpoch %05d: learning rate is minimum %s. Stopping training.' % (epoch, new_lr))
                            self.model.stop_training = True

                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0


                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


#----------------------------------------------------------------------------------------------------

class ModelCheckpointProtected(callbacks.Callback):
    """Same as ModelCheckpoint but makes sure the model is saved to disk even if SIGINT is received.
    It also reads custom history file to support arbitrary monitor values. """

    def __init__(self, history_path, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointProtected, self).__init__()
        self.history_path = history_path
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best_op = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
            self.best_op = lambda x: np.min(x)
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_op = lambda x: np.max(x)
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best_op = lambda x: np.max(x)
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best_op = lambda x: np.min(x)
                self.best = np.Inf

    # def on_train_begin(self, logs=None):
    #     filepath = self.filepath.format(epoch=0, **logs)
    #     self.model.save(filepath, overwrite=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                # current = logs.get(self.monitor)
                df = pd.read_csv(self.history_path)
                current = df.loc[len(df) - 1, self.monitor]
                self.best = self.best if len(df.loc[df[self.monitor].notnull(), :]) <= 1 else self.best_op(df.loc[df[self.monitor].notnull(), self.monitor].values[:-1])  # checks best score from history (useful for resuming) excluding current metric

                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        try:

                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)

                        except KilledException as e:

                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)

                            raise KilledException()

                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve (current: %0.5f, best: %0.5f, monitor_op: %s, best_op: %s)' %
                                  (epoch, self.monitor, current, self.best, str(self.monitor_op), str(self.best_op)))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                try:

                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

                except KilledException as e:

                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

                    raise KilledException()

#----------------------------------------------------------------------------------------------------

class EvalCustomMetric(callbacks.Callback):
    """Callback that measures performance on entire set"""

    def __init__(self, log_path, dataset, fn_performances, fn_labels, tags, on_epoch=False, use_multiprocessing=False,
                 workers=1, alternative_output=None, smooth_window=0, pred_on_batch=False, steps=None):
        self.on_epoch = on_epoch
        self.alternative_output = alternative_output
        self.dataset = dataset
        self.fn_performances = fn_performances
        self.fn_labels = fn_labels
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers
        self.tags = tags
        self.log_path = log_path
        self.smooth_window = smooth_window
        self.pred_on_batch = pred_on_batch
        self.steps = steps
        super(EvalCustomMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if self.on_epoch:
            if self.pred_on_batch:
                val = self.validate_on_batch()
            else:
                val = self.validate()
            self.update_log(val)

    def on_train_end(self, logs=None):
        if self.pred_on_batch:
            val = self.validate_on_batch()
        else:
            val = self.validate()
        self.update_log(val)

    def validate(self):

        model = self.model

        y_pred = model.predict_generator(
            generator=self.dataset,
            steps=len(self.dataset) if self.steps is None else np.min([len(self.dataset), self.steps]),
            max_queue_size=10,
            workers=self.workers,
            use_multiprocessing=self.use_multiprocessing,
            verbose=0  # TODO log errors only
        )
        y_true = self.fn_labels(self.dataset)

        # Metric
        metrics = {}
        for i in range(len(self.fn_performances)):
            # print('y_pred is: {y_pred}'.format(y_pred=y_pred), flush=True)
            # print('y_true is: {y_true}'.format(y_true=y_true), flush=True)
            # print('fn_performance is: {fn_performance}'.format(fn_performance=self.fn_performances[i]), flush=True)

            metrics[self.tags[i]] = self.fn_performances[i](y_true, y_pred)

        return metrics

    def validate_on_batch(self):

        model = self.model

        # Metric
        metrics = {}
        for i in range(len(self.fn_performances)):

            results = []
            steps = len(self.dataset) if self.steps is None else np.min([len(self.dataset), self.steps])

            for x, y_true in self.dataset:
                y_pred = model.predict_on_batch(x)
                results.append(self.fn_performances[i](y_true, y_pred))

                steps -= 1
                if steps <= 0:
                    break

            metrics[self.tags[i]] = np.mean(results)

        return metrics

    def update_log(self, val):

        # Read log
        df = pd.read_csv(self.log_path)

        # For each metric
        for tag, metric_value in val.items():

            # Column name
            if tag not in df.columns:
                df[tag] = np.nan

            # Add latest value
            df.loc[len(df) - 1, tag] = metric_value

        # Rolling mean
        if self.smooth_window > 0:
            for tag in self.tags:
                try:
                    # df[tag + '_smooth'] = df[tag].fillna(method='ffill').rolling(self.smooth_window, center=False).mean()
                    df[tag + '_smooth'] = df[tag].fillna(method='ffill').rolling(self.smooth_window, center=False).median()
                except Exception as e:
                    print('Failed to produce rolling mean with exception: {e}'.format(e=e), flush=True)

        # Store
        df.to_csv(self.log_path)

#----------------------------------------------------------------------------------------------------


class CommonMetric(callbacks.Callback):

    def __init__(self, log_path, metric_tags, new_metric, fn_metric):
        self.log_path = log_path
        self.metric_tags = metric_tags
        self.new_metric = new_metric
        self.fn_metric = fn_metric
        super(CommonMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        # Read log
        df = pd.read_csv(self.log_path)

        # Column name
        if self.new_metric not in df.columns:
            df[self.new_metric] = np.nan

        # Compute new metric
        df.loc[len(df) - 1, self.new_metric] = self.fn_metric(df.loc[len(df) - 1, self.metric_tags])

        # Store
        df.to_csv(self.log_path)

#----------------------------------------------------------------------------------------------------

def conv_op(input_layer, n_filters, f_size=(3, 3), stride=1, bn=True, activation='relu', padding='same', name=None,
            l2_factor=None, use_separable_conv=False, bn_momentum=0.99):
    """
    Convolutional operation.

    Args:
        input_layer: input layer (Keras layer).
        n_filters: number of filters to convolve with.
        f_size: size of filter. For example 3 or (3, 3).
        stride: stride of convolution.
        bn: True to enable batch normalization.
        activation: non-linearity to use. For example 'relu', 'linear' or 'lrelu' for leaky-relu.
        padding: padding 'same' or 'valid'.
        name: unique name of layer.
        l2_factor: factor for L2 regularization.
        use_separable_conv (bool): True to use separable convolutions instead of regular ones.

    Returns: Keras layer.

    """

    # Regularization
    if l2_factor is not None:
        l2_reg = regularizers.l2(l2_factor)
    else:
        l2_reg = None

    # Conv
    if use_separable_conv:
        x = layers.SeparableConv2D(filters=n_filters, kernel_size=f_size, strides=stride, padding=padding, depth_multiplier=1,
                           activation='linear', name=name, depthwise_regularizer=l2_reg, pointwise_regularizer=l2_reg,
                           bias_regularizer=l2_reg, kernel_initializer='he_uniform')(input_layer)
    else:
        x = layers.Conv2D(filters=n_filters, kernel_size=f_size, strides=stride, padding=padding,
                        activation='linear', name=name, kernel_regularizer=l2_reg, bias_regularizer=l2_reg,
                        kernel_initializer='he_uniform')(input_layer)



    # Batch norm
    if bn:
        x = layers.BatchNormalization(axis=-1, momentum=bn_momentum)(x)

    # Activation
    if activation == 'lrelu':
        x = layers.LeakyReLU()(x)
    else:
        x = layers.Activation(activation)(x)

    return x

#----------------------------------------------------------------------------------------------------

def dense_op(input_layer, n_units, bn=True, activation='relu', name=None, l2_factor=None, bn_momentum=0.99):
    """
    Dense operation.

    Args:
        input_layer: input layer (Keras layer).
        n_units: number of units in the dense layer.
        bn: True to enable batch normalization.
        activation: non-linearity to use. For example 'relu', 'linear' or 'lrelu' for leaky-relu.
        name: unique name of layer.
        l2_factor: factor for L2 regularization.

    Returns: Keras layer.

    """

    # Names (trouble with other dense layers otherwise)
    if name is None:
        dense_name = None
        bn_name = None
        act_name = None
    else:
        dense_name = '%s' % name
        bn_name = 'bn_%s' % name
        act_name = 'act_%s' % name

    # Regularization
    if l2_factor is not None:
        l2_reg = regularizers.l2(l2_factor)
    else:
        l2_reg = None

    # Op
    x = layers.Dense(units=n_units, activation='linear', name=dense_name, kernel_regularizer=l2_reg,
                     bias_regularizer=l2_reg, kernel_initializer='he_uniform')(input_layer)

    # Batch norm
    if bn:
        x = layers.BatchNormalization(axis=-1, name=bn_name, momentum=bn_momentum)(x)

    # Activation
    if activation == 'lrelu':
        x = layers.LeakyReLU(name=act_name)(x)
    else:
        x = layers.Activation(activation, name=act_name)(x)

    return x

#----------------------------------------------------------------------------------------------------
