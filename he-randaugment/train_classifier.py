"""
This module trains a supervised multiclass classifier.
"""

# import utils.set_random_seed  # sets random seeds
from data_generator import SupervisedGenerator, SupervisedSequence, SupervisedSequenceSingle
from dl import pred_model

import pandas as pd
import dl
from os.path import join, exists, basename, dirname, splitext
import numpy as np
import argparse
from glob import glob
import gc
import metrics
import time
import random
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------------------


def train_classifier(dataset_dir, output_dir, n_epochs, batch_size, augmenter, lr, network_tag,
                     patch_size=(128, 128, 3), workers=1, color_space='rgb', test_only=False, test_dataset_dir=None,
                     randaugment=True, rand_m=5,  rand_n=1,ra_type=None, v1_type=None, v2_type=None, t1_type=None, t2_type=None):
    """
    This function orchestrates the training of a neural network on a supervised dataset.

    Args:
        dataset_dir (str): directory where the training and validation data live (training_x.npy, training_y.npy, etc.).
        output_dir (str): directory where the results will be stored.
        n_epochs (int): number of training epochs to run.
        batch_size (int): number of samples per batch.
        lr (float): learning rate.
        code_size (int): size of the embedding (code).
        tag (str): result folder name.

    """

    if (not test_only) and (not exists(join(output_dir, 'training_finished.txt'))):

        # Randomize starting time
        print('Waiting a few seconds ...', flush=True)
        time.sleep(random.randint(1, 180))

        # Training set
        print('Loading training set ...', flush=True)
        training_gen = SupervisedGenerator(
            x_path=join(dataset_dir, 'training_x.npy'),
            y_path=join(dataset_dir, 'training_y.npy'),
            batch_size=batch_size,
            augmenter=augmenter,
            one_hot=True,
            color_space=color_space,
            randaugment=randaugment, 
            rand_m=rand_m,  
            rand_n=rand_n,
            ra_type=ra_type
        )
      
        # Validation set
        print('___0___',join(dataset_dir, 'validation_x.npy'),'___1___',join(dataset_dir,'test_'+v1_type+'_x.npy'),'___2___',join(dataset_dir,'test_'+v2_type+'_x.npy'))
        print('Loading validation set ...', flush=True)
        
        
        
        validation_gen = SupervisedSequence(
            x_path=[join(dataset_dir, 'validation_x.npy'),join(dataset_dir,'test_'+v1_type+'_x.npy'),join(dataset_dir,'test_'+v2_type+'_x.npy')],
            y_path=[join(dataset_dir, 'validation_y.npy'),join(dataset_dir,'test_'+v1_type+'_y.npy'),join(dataset_dir,'test_'+v2_type+'_y.npy')],
            batch_size=batch_size,
            one_hot=True,
            color_space=color_space
        )

        # Create model
        print('Building model ...', flush=True)
        model = build_classifier_model(input_shape=patch_size, n_classes=training_gen.get_n_classes(), lr=lr, network_tag=network_tag)

        # Callback
        patience = 4
        callbacks = [
            dl.EvalCustomMetric(
                log_path=join(output_dir, 'history.csv'),
                dataset=validation_gen,
                fn_performances=[metrics.log_loss_fn, metrics.log_loss_weighted_fn, metrics.accuracy_fn, metrics.accuracy_weighted_fn],
                fn_labels=lambda x: x.get_all_labels(one_hot=True),
                tags=['val_loss', 'val_loss_weighted', 'val_categorical_accuracy', 'val_categorical_accuracy_weighted'],
                on_epoch=True,
                use_multiprocessing=False,
                workers=1,  # disabled due to OverflowError: cannot serialize a bytes object larger than 4 GiB
                smooth_window=patience
            )
        ]

        # Train model
        print('Training model ...', flush=True)
        dl.fit_model(
            train_gen=training_gen,
            val_gen=None,
            model=model,
            n_epochs=n_epochs,
            output_dir=output_dir,
            workers=workers,
            loss_list=['loss', 'val_loss', 'val_loss_weighted', 'val_loss_weighted_smooth'],
            metric_list=['categorical_accuracy', 'val_categorical_accuracy', 'val_categorical_accuracy_weighted', 'val_categorical_accuracy_weighted_smooth'],
            extra_callbacks=callbacks,
            monitor='val_loss_weighted_smooth',
            mode='min',
            min_lr=1e-5,
            patience=patience,
            steps_per_epoch=int(100000/batch_size),
            warm_up_model=True if 'pretrained' in network_tag else False
        )

        del training_gen
        del validation_gen
        gc.collect()  # Force releasing memory (otherwise mem error in cluster)
    else:
        print('Training phase not required or already completed.', flush=True)

    # Test model
    model_path = join(output_dir, 'checkpoint.h5')
    if exists(model_path):

        # Normal predictions
        for x_test_path in glob(join(dataset_dir, 'test_*_x.npy')):
            
            test_id = splitext(basename(x_test_path))[0].split('_')[-2]
            output_path = join(output_dir, 'preds_{test_id}.npy'.format(test_id=test_id))
            if not exists(output_path):

                print('Loading test set ({p})...'.format(p=x_test_path), flush=True)
                test_gen = SupervisedSequenceSingle(
                    x_path=join(dataset_dir, 'test_{test_id}_x.npy'.format(test_id=test_id)),
                    y_path=join(dataset_dir, 'test_{test_id}_y.npy'.format(test_id=test_id)),
                    batch_size=batch_size,
                    one_hot=True,
                    color_space=color_space
                )

                print('Testing model ...', flush=True)
                pred_model(
                    data_seq=test_gen,
                    model_path=model_path,
                    output_path=output_path,
                    steps=None
                )

                del test_gen  # Need to delete it explicitly to save memory
                gc.collect()  # Force releasing memory (otherwise mem error in cluster)

            else:
                print('Prediction already existing in disk: {f}'.format(f=output_path), flush=True)


        
        output_path_val = join(output_dir, 'preds_validation.npy')
        if not exists(output_path_val):

            print('Loading test set ({p})...'.format(p=x_test_path), flush=True)
            val_gen = SupervisedSequenceSingle(
                x_path=join(dataset_dir, 'validation_x.npy'),
                y_path=join(dataset_dir, 'validation_y.npy'),
                batch_size=batch_size,
                one_hot=True,
                color_space=color_space
            )

            print('Testing model ...', flush=True)
            pred_model(
                data_seq=val_gen,
                model_path=model_path,
                output_path=output_path_val,
                steps=None
            )

            del val_gen  # Need to delete it explicitly to save memory
            gc.collect()  # Force releasing memory (otherwise mem error in cluster)

        else:
            print('Prediction already existing in disk: {f}'.format(f=output_path_val), flush=True)


#----------------------------------------------------------------------------------------------------

def check_completed_epochs(output_dir):

    epochs_run = 0
    if exists(join(output_dir, 'last_epoch.h5')) and exists(join(output_dir, 'history.csv')):
        df = pd.read_csv(join(output_dir, 'history.csv'))
        epochs_run = len(df)

    return epochs_run


#----------------------------------------------------------------------------------------------------

def build_classifier_model(input_shape, n_classes, lr, network_tag):
    """
    Builds a neural network that distinguishes between tumor or healthy images.

    Args:
        input_shape: shape of image with channels last, for example (128, 128, 3).
        lr (float): learning rate.
        tag (str): model tag.

    Returns: compiled Keras model.

    """

    # Input
    input_x = dl.layers.Input(input_shape)

    if network_tag == 'basic':
        # Define classifier
        l2_factor = 1e-6
        x = dl.conv_op(input_x, 32, stride=1, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 64, stride=2, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 64, stride=1, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 128, stride=2, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 128, stride=1, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 256, stride=2, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 256, stride=1, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 512, stride=2, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.conv_op(x, 512, stride=1, bn=True, activation='lrelu', padding='valid', l2_factor=l2_factor)
        x = dl.layers.GlobalAveragePooling2D()(x)
        x = dl.layers.Dropout(0.5)(x)
        x = dl.dense_op(x, 512, bn=True, activation='lrelu', name='dense1', l2_factor=l2_factor)
        x = dl.dense_op(x, n_classes, bn=False, activation='softmax', name='output', l2_factor=l2_factor)


    else:
        raise Exception('Unknown network architecture {n}'.format(n=network_tag))

    # Compile
    model = dl.models.Model(inputs=input_x, outputs=x)
    model.compile(
        optimizer=dl.optimizers.Adam(lr=lr),
        loss=dl.losses.categorical_crossentropy,
        metrics=[dl.metrics.categorical_accuracy]
    )

    # model.summary()

    return model
