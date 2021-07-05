#%% Import libraries

import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras, math
from tensorflow.test import gpu_device_name
from sklearn.model_selection import train_test_split
from model import tf_model, dataset_class
import time

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = 'large'

########################
## PARAMETERS ##########
########################
parser = argparse.ArgumentParser(prog = '\n>> script_name.py --argname argument \n',
        description = 'The program runs DNN model training for object detection to locate the banana in the pictures provided.\n Please provide train_folder and test_folder hierarchy as described in the parameter specification')

parser.add_argument('--train_folder', type=str, nargs='?', default=os.path.join('banana-detection', 'bananas_train'), 
                        help='Path to the folder containing the training "images" folder and the "label.csv" file')
parser.add_argument('--test_folder', type=str, nargs='?', default=os.path.join('banana-detection', 'bananas_test'), 
                        help='Path to the folder containing the test "images" folder and the "label.csv" file')
parser.add_argument('--ws', type=int, nargs='?', default=50, help='Window size')
parser.add_argument('--stride', type=int, nargs='?', default=25, help='Stride value')
parser.add_argument('--overlap', type=float, nargs="?", default=0.3, 
        help='Set to value between 0 and 1. If preprocessing script is called it sets the minimum amount of box overlap needed to save a patch as containing the object (label=1)')
parser.add_argument('--model_folder', type=str, nargs='?', default="model", help='Path to the folder used to save the model')
parser.add_argument('--reg', type=lambda x: not (str(x).lower() in ['false','0', 'no']), nargs="?", default=True, help='Set to True to use regressive smoothing model, False for classifier only')
parser.add_argument('--augment', type=lambda x: (str(x).lower() in ['true','1', 'yes']), nargs="?", default=False, help='Set to True to force preprocessing with the data augmentation script')
parser.add_argument('--test', type=lambda x: not (str(x).lower() in ['false','0', 'no']), nargs="?", default=True, help='Set to True to skip training, and only evaluate the model on the test set. It needs an existing model to work')
########################
########################
########################


if __name__ == '__main__':

    ### Parse input arguments
    args = parser.parse_args()

    print("Running prediction with options:\n", args)

    train_folder = os.path.join('banana-detection', 'bananas_train')#args.train_folder#os.path.join('banana-detection', 'bananas_train')

    window_size = args.ws#50 # 70
    stride = args.stride#25 # 35
    overlap = args.overlap#0.3

    test = args.test#False
    regression = args.reg#True
    augment = args.augment

    # Check if gpu is available
    if gpu_device_name():
        print('Default GPU Device:{}'.format(gpu_device_name()))
    else:
        print("\n***NO GPU DETECTED.....")

    if regression:
        model_folder = os.path.join(args.model_folder, "classifier_"+str(window_size)+"_reg")
    else:
        model_folder = os.path.join(args.model_folder, "classifier_"+str(window_size))
    model_code = "_"+str(window_size)

    if not test:

        if (not os.path.exists(os.path.join(train_folder, "label_processed"+model_code+".csv"))) or augment:
            print("***Processed images folder and label file not detected, running preprocessing...")
            print("***Processing folder", train_folder, "...")
            subprocess.run(["augmentation.exe", train_folder, str(window_size), 
                                str(stride), str(overlap), 
                                "train", str(15)], stdout=subprocess.PIPE).stdout.decode('utf-8')
            print("***All done!")

        train_head = pd.read_csv(os.path.join(train_folder, "label_processed"+model_code+".csv"), index_col='img_name')
        print("Label distribution:\n", train_head.groupby("label").count()['xmin'], "\n")

	    # Equalize label diversity of training samples
        n_drops = len(train_head[train_head['label']==0]) - len(train_head[train_head['label']==1])
        drop_indices = np.random.choice(train_head[train_head['label']==0].index, n_drops, replace=False)
        train_head = train_head.drop(drop_indices)
        print("Label distribution after equalization:\n", train_head.groupby("label").count()['xmin'], "\n")

	    # Split in train and validation
        train_head, val_head = train_test_split(train_head, 
                                            test_size=0.1, stratify=train_head['label'],
                                            random_state=423, shuffle=True)
    
        print("Label distribution after equalization TRAINING SET:\n", train_head.groupby("label").count()['xmin'], "\n")
        print("Label distribution after equalization VALIDATION SET:\n", val_head.groupby("label").count()['xmin'], "\n")

        #*********************************************************************************************#
        #*********************************************************************************************#

        start = time.time()
        # Create dataset class
        train_ds = dataset_class.BananaDataGenerator(train_head, os.path.join(train_folder, "images_processed"+model_code))
        tr = time.time()
        val_ds = dataset_class.BananaDataGenerator(val_head, os.path.join(train_folder, "images_processed"+model_code))
        v = time.time()

        batch_size = 20
        num_epochs = 30
        train_steps = len(train_head)//batch_size
        val_steps = len(val_head)//batch_size
        input_shape = train_ds.imshape

        # Create tf pipeline for the two dataset
        if regression:
            train_pipe = train_ds.regression_pipe(batch_size= batch_size)
            tr_p = time.time()
            val_pipe = val_ds.regression_pipe(batch_size = batch_size)
            v_p = time.time()
        else:
            train_pipe = train_ds.pipe(batch_size= batch_size)
            tr_p = time.time()
            val_pipe = val_ds.pipe(batch_size = batch_size)
            v_p = time.time()
        print("\nTimings:\n", "\nTrain Dataset ->", tr - start, "\nVal Dataset ->", v - tr, 
                            "\nTrain Pipeline ->", tr_p - v, "\nVal Pipeline ->", v_p - tr_p, 
                            "\nTotal ->", v_p - start)

        #print(next(iter(train_pipe))[1])

        # Define useful callbacks
        def scheduler(epoch, lr):
            if epoch < 4: return lr
            else: return lr * math.exp(-0.1)

        #printlr_callback = tf_model.printlearningrate()
        #lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=0)
        lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Define optimizer
        opt = keras.optimizers.Adam(learning_rate=0.001)

        # Create model
        model = tf_model.BananaDetectionModel().Classifier_with_regression(input_shape)
        model.summary()

        if regression:
            model.compile(optimizer = opt, 
                            loss = {
                                'classifier_output' : 'binary_crossentropy',
                                'regression_output' : 'mse'},
                            metrics = {
                                'classifier_output' : 'accuracy',
                                'regression_output' : 'mae'})
        else:
            model.compile(optimizer = opt, loss = ['binary_crossentropy'], metrics = ['accuracy'])

        history = model.fit(train_pipe, epochs=num_epochs,
	    						steps_per_epoch = train_steps,
	    						validation_data = val_pipe,
	    						validation_steps = val_steps,
	    						callbacks = [lr_callback, early_stop_callback], 
                                shuffle=True)

        #Save model
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        model.save(model_folder)
    
#%%        ##Plotting
        fontsize=12
        if regression:
            # Plot loss
            plt.figure(figsize=(10,5))
            plt.plot(history.history['classifier_output_loss'], label='Train loss')
            plt.plot(history.history['val_classifier_output_loss'], label='Val loss')
            plt.legend(fontsize=fontsize); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Classifier Branch Loss', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "classifier_loss.png"))
            # Regression Loss
            plt.figure(figsize=(10,5))
            plt.plot(history.history['regression_output_loss'], label='Train loss')
            plt.plot(history.history['val_regression_output_loss'], label='Val loss')
            plt.legend(fontsize=fontsize); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Regression Branch Loss', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "regression_loss.png"))
            # Plot accuracy
            plt.figure(figsize=(10,5))
            plt.plot(history.history['classifier_output_accuracy'], label='Train accuracy')
            plt.plot(history.history['val_classifier_output_accuracy'], label='Val accuracy')
            plt.legend(fontsize=fontsize); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Classifier Branch Accuracy', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "classifier_accuracy.png"))
            # Regression MAE
            plt.figure(figsize=(10,5))
            plt.plot(history.history['regression_output_mae'], label='Train MAE')
            plt.plot(history.history['val_regression_output_mae'], label='Val MAE')
            plt.legend(fontsize=fontsize); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Regression Branch Mean Absolute Error', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "regression_mae.png"))
            # Plot Learning rate
            plt.figure(figsize=(10,5))
            plt.plot(history.history['lr'], label='Learning rate')
            plt.legend(fontsize=fontsize); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Learning rate', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "lr.png"))
        elif (not regression):
            # Plot loss
            plt.figure(figsize=(10,5))
            plt.plot(history.history['loss'], label='Train loss')
            plt.plot(history.history['val_loss'], label='Val loss')
            plt.legend(fontsize=fontsize); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Loss', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "classifier_loss.png"))
	        # Plot accuracy
            plt.figure(figsize=(10,5))
            plt.plot(history.history['accuracy'], label='Train accuracy')
            plt.plot(history.history['val_accuracy'], label='Val accuracy')
            plt.legend(); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Accuracy', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "classifier_accuracy.png"))
	        # Plot Learning rate
            plt.figure(figsize=(10,5))
            plt.plot(history.history['lr'], label='Learning rate')
            plt.legend(); #plt.grid()
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel('Learning rate', fontsize=fontsize)
            plt.savefig(os.path.join(model_folder, "lr.png"))


#%%    ##Test

    if not os.path.exists(model_folder):
        print("***No corresponding model exists in the given model_folder...exiting")
        exit(1)
    
    # Load Model
    model = keras.models.load_model(model_folder)
    test_batch_size = 20

    test_folder = args.test_folder

    if (not os.path.exists(os.path.join(test_folder, "label_processed"+model_code+".csv"))) or augment:
        print("***Processed images folder and label file not detected, running preprocessing...")
        print("***Processing folder", test_folder, "...")
        subprocess.run(["augmentation.exe", test_folder, str(window_size), 
                            str(stride), str(overlap), "no", 0], stdout=subprocess.PIPE).stdout.decode('utf-8')
        print("***All done!")

    # Load Test dataset header
    test_head = pd.read_csv(os.path.join(test_folder, "label_processed"+model_code+".csv"), index_col='img_name')
    # Dataset class generator object
    test_ds = dataset_class.BananaDataGenerator(test_head, os.path.join(test_folder, "images_processed"+model_code))

    if regression:
        test_pipe = test_ds.regression_pipe(batch_size=test_batch_size, repeat=False)
    else:
        test_pipe = test_ds.pipe(batch_size=batch_size, repeat=False)

    results = model.evaluate(test_pipe)

    print("Test results:\n", model.metrics_names,"\n", results)