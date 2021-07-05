#%% Import libraries

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.test import gpu_device_name
from model import dataset_class
import matplotlib.patches as patches

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['legend.fontsize'] = 'large'
import time


########################
## PARAMETERS ##########
########################
parser = argparse.ArgumentParser(prog = '\n>> script_name.py --argname argument \n',
        description = 'The program runs object detection to locate the banana in the pictures provided in the input_folder argument.\n Please provide input_folder hierarchy as described in the parameter specification')

parser.add_argument('--input_folder', type=str, nargs='?', default=os.path.join('banana-detection', 'bananas_test'), 
                        help='Path to the folder containing the "images" folder and the "label.csv" file')
parser.add_argument('--ws', type=int, nargs='?', default=50, help='Window size')
parser.add_argument('--stride', type=int, nargs='?', default=25, help='Stride value')
parser.add_argument('--model_folder', type=str, nargs='?', default="model", help='Path to the folder containing the saved models')
parser.add_argument('--reg', type=lambda x: not (str(x).lower() in ['false','0', 'no']), nargs="?", default=True, help='Set to True to use regressive smoothing model, False for classifier only')
parser.add_argument('--smoothing_factor', type=float, nargs="?", default=0.6, help='Set to value between 0 and 1. It controls the amount of smoothing of the bounding box')
parser.add_argument('--detection_thr', type=float, nargs="?", default=0.9, help='Set to value between 0 and 1. It controls the threshold value above which a detection is considered valid')

########################
########################
########################


if __name__ == '__main__':


    ### Parse input arguments
    args = parser.parse_args()

    print("Running prediction with options:\n", args)

    window_size = args.ws
    stride = args.stride
    regression = args.reg

    test_folder = args.input_folder

    if regression:
        model_folder = os.path.join(args.model_folder, "classifier_"+str(window_size)+"_reg")
        output_folder = os.path.join(test_folder, "images_detected_"+str(window_size)+"_reg")
    else:
        model_folder = os.path.join(args.model_folder, "classifier_"+str(window_size))
        output_folder = os.path.join(test_folder, "images_detected_"+str(window_size))
    model_code = "_"+str(window_size)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Check if gpu is available
    if gpu_device_name():
        print('Default GPU Device:{}'.format(gpu_device_name()))
    else:
        print("\n***NO GPU DETECTED.....")


    # Load Model
    model = models.load_model(model_folder)

    # Load Test dataset header
    test_head = pd.read_csv(os.path.join(test_folder, "label.csv"), index_col='img_name')
    # Dataset class generator object
    test_ds = dataset_class.BananaDataGenerator(test_head, os.path.join(test_folder, "images"))#, extension='.JPG')
    # Save original bounding boxes as proper object for future use
    orig_boxes = dataset_class.BoundingBoxes(test_ds.bboxes)

    print("***Data loaded...begin processing")
    
    for idx in range(len(test_head)):
        # Load tf pipeline for original image
        patch_pipe = test_ds.split_image_pipe(test_ds.fnames[idx], ws = window_size, stride = stride)

        start = time.time()
        # Predict labels
        pred = model.predict(patch_pipe)
        end = time.time()
        # Load them into proper object for cleaner use
        preds = dataset_class.PredictionLabels(pred)
        mask = preds.threshold(args.detection_thr) # Filter best detections

        # Select the predicted boxes from list of boxes corresponding to each patch
        test_ds.patch_list = test_ds.patch_list[1:]
        # Locate them in proper object class
        boxes = dataset_class.BoundingBoxes(test_ds.patch_list[mask])

        print(test_ds.fnames[idx], "\tPrediction time:", end-start, "\tNumber of detected subboxes:", len(boxes.boxes))

        ### Plotting
        fig, ax = plt.subplots()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.imshow(test_ds.load_image(test_ds.fnames[idx]))
        ax.text(5, 7, "Box Center,    [Detection Probability, Score]", bbox=dict(fill=True, facecolor='C1', edgecolor=None, linewidth=2))
        if boxes.boxes.size!=0:
            # Plot original box GREEN
            rect = patches.Rectangle((orig_boxes[idx][0], orig_boxes[idx][1]), 
                                    orig_boxes.width[idx], 
                                    orig_boxes.height[idx], 
                                    linewidth=3.5, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            for i,box in enumerate(boxes):
                #print(box, boxes.center[i], "\t[Detection Probability, Score]:", preds[i])
                text = str(boxes.center[i])+",   "+str(preds[i])
                ax.text(5, 17+i*10, text, bbox=dict(fill=None, edgecolor="C1", linewidth=2), color='black', fontweight='bold')
                rect = patches.Rectangle((box[0], box[1]), window_size, window_size, linewidth=1.5, edgecolor='C1', facecolor='none')
                ax.add_patch(rect)

            # Plot predicted big box
            #big_box = boxes.bigbox()
            #rect = patches.Rectangle((big_box[0], big_box[1]), 
            #                        big_box.width, 
            #                        big_box.height, linewidth=2, edgecolor='b', facecolor='none')
            #ax.add_patch(rect)

            # Plot smoothed predicted box
            if regression:
                boxes = boxes.smooth_detection(preds.probs, args.smoothing_factor)
                big_box = boxes.bigbox()
                rect = patches.Rectangle((big_box[0], big_box[1]), 
                                        big_box.width, 
                                        big_box.height, linewidth=3.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        
        # Save images
        plt.savefig(os.path.join(output_folder, test_ds.fnames[idx]), bbox_inches='tight', pad_inches=0)
        if idx%100==0:
            plt.show()
        plt.close()