
#%% Importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import float32, expand_dims
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import img_to_array

class PredictionLabels():
    """
    Class including utilities for processing network prediction output
    """
    def __init__(self, pred_labels : np.ndarray) -> None:
        if len(pred_labels[0]) >= 2:
            self.labels = np.array(pred_labels[0]).squeeze()
            self.categorical_labels = np.array([1 if pred>0.5 else 0 for pred in pred_labels[0]]).squeeze()
            self.probs = np.array(pred_labels[1]).squeeze()
        else:
            self.labels = np.array(pred_labels).squeeze()
            self.categorical_labels = np.array([1 if pred>0.5 else 0 for pred in pred_labels]).squeeze()
    
    def __getitem__(self, sliced) -> np.ndarray:
        if hasattr(self, 'probs'):
            if isinstance(sliced, int):
                return [self.labels[sliced], self.probs[sliced]]
            else:
                return zip(self.labels[sliced], self.probs[sliced])
        else:
            return self.labels[sliced]

    def apply_mask(self, mask : np.ndarray, save_mask : bool = True) -> None:
        """
        Applies filtering mask to class object
        """
        self.labels = self.labels[mask]
        self.categorical_labels = self.categorical_labels[mask]
        if hasattr(self, 'probs'):
            self.probs = self.probs[mask]
        if save_mask:
            self.mask = mask


    def threshold(self, threshold : float = 0.95, apply_mask : bool = True) -> np.ndarray:
        """
        Filters the class objects retaining only the indexes corresponing to labels objects > threshold
        and outputs the corresponding mask for matching with other objects
        """
        mask = np.zeros_like(self.labels, dtype=bool)
        for i in np.arange(len(self.labels)):
            if self.labels[i] > threshold: mask[i]=True
            else: mask[i]=False
        if len(mask[mask==True])<2:
            # If no detection, recursively try smaller values of threshold
            print("***No detection above threshold", threshold)
            threshold = threshold-threshold*0.1
            if threshold<0.5:
                return mask
            print("***Trying threshold", threshold)
            mask = self.threshold(threshold, apply_mask = False)
        if apply_mask:
            self.apply_mask(mask)
        return mask
        


class BoundingBoxes():
    """
    Class including useful utilities for bounding boxes
    """
    def __init__(self, boxes : np.ndarray) -> None:
        self.boxes = boxes#.squeeze()
        if boxes.size//4 > 1 or len(boxes.shape) >= 2:
            self.height = np.array([box[3] - box[1] for box in boxes])
            self.width = np.array([box[2] - box[0] for box in boxes])
            self.center = np.array([[box[0]+w//2, box[1]+h//2] for box,w,h in zip(boxes, self.height, self.width)])
        elif boxes.size!=0:
            self.height = boxes[3] - boxes[1]
            self.width = boxes[2] - boxes[0]
            self.center = np.array([boxes[0]+self.width//2, boxes[1]+self.height//2])

    def __getitem__(self, sliced) -> np.ndarray:
        return self.boxes[sliced]#BoundingBoxes(self.boxes[sliced])

    def bigbox(self):
        """
        Approximate multiple boxes with one big one containing all of them
        """
        if self.boxes.size//4 > 1:#len(self.boxes.shape) >= 2 or self.boxes.shape[0] > 0:
            return BoundingBoxes(np.array([min(self.boxes[:,0]), min(self.boxes[:,1]),
                                            min(self.boxes[:,0]) + (max(self.boxes[:,2])-min(self.boxes[:,0])),
                                            min(self.boxes[:,1]) + (max(self.boxes[:,3])-min(self.boxes[:,1]))]))
        else: 
            self.boxes = self.boxes.squeeze()
            return self

    def smooth_detection(self, pred_probs : np.ndarray, smoothing_factor : float = 0.1):
        """
        Smoothens the position of the boxes proportionally to the smoothing_factor. 
        The boxes are moved towards their centroid weighted on the scoring pred_probs 
        associated with each of them
        """
        if self.boxes.size//4 > 1:# or self.boxes.shape[0] > 0:
            if len(pred_probs)!=len(self.boxes):
                print("Size of the bboxes probabilities don't match!")
                return
            else:
                # Normalize prediction probabilities
                pred_probs = pred_probs/np.sum(pred_probs)
                # Compute Centroid with normalized detection probabilities as weights
                centroid = np.array([np.sum(self.center[:,0]*pred_probs), np.sum(self.center[:,1]*pred_probs)])
                # Move the centers closer to the centroid proportionally to smoothing_factor 
                n_centers = np.array([MovePoint(center, centroid, smoothing_factor) for center in self.center])
                n_boxes = np.array(list(map(BoxfromCenter, n_centers, self.width, self.height)))
                return BoundingBoxes(n_boxes)#n_boxes, n_centers, pred_probs, centroid
        else:
            return self


def BoxfromCenter(center : np.ndarray, width : int, height : int):
    """
    Computes new box from center point and box dimensions
    """
    return np.array([center[0]-width//2, center[1]-height//2, center[0]+width//2, center[1]+height//2])

def MovePoint(start_p : np.ndarray, end_p : np.ndarray, factor : float):
    """
    Moves start_p point towards end_p point proportionally to factor
    """
    move = lambda x_1,x_2 : (x_1 + int(factor*(x_2 - x_1)))
    return np.array([move(start_p[0], end_p[0]), move(start_p[1], end_p[1])])



class BananaDataGenerator():
    """
    Data generator for the object detection image dataset
    """
    def __init__(self, df : pd.DataFrame, imgfolderpath : str, extension : str ='.png') -> None:

        self.imgfolder = imgfolderpath
        self.fextension = extension
        self.fnames = np.array(df.index)
        self.labels = df['label'].astype(np.float32)
        if 'reg_label' in df.columns:
            self.reg_labels = df['reg_label'].astype(np.float32)
        self.bboxes = np.array(df[['xmin', 'ymin', 'xmax', 'ymax']].values)
        #Get image shape
        self.imshape = self.load_image(self.fnames[0]).shape

    def load_image(self, fname : str) -> np.ndarray:
        """
        Image loader function
        """
        # Decode for tensorflow strings encoding
        if isinstance(fname, bytes):
            fname = fname.decode()
        if os.path.splitext(fname)[-1] != self.fextension: # Add extention if not present
            fname = fname+self.fextension

        img = img_to_array(plt.imread(os.path.join(self.imgfolder, fname)), dtype=np.float32)
        ### Possible addition of splitting or resizing
        return img

    def generator(self):
        """
        Dataset Generator
        """
        for i, name in enumerate(self.fnames):
            yield self.load_image(name), np.array(self.labels[i])
    
    def regression_generator(self):
        """
        Dataset Generator with regression label
        """
        for i, name in enumerate(self.fnames):
            yield self.load_image(name), {'classifier_output' : np.array(self.labels[i]), 
                                           'regression_output' : np.array(self.reg_labels[i])}

    def split_image_generator(self, fname : str, ws : int, stride : int):
        """
        Generator for image splitting into patches
        """
        image = self.load_image(fname)
        self.patch_list = np.zeros(shape=([1,4]), dtype= np.int64)# Placeholder for patch index dictionary
        for y in range(0, image.shape[0] - ws, stride):
            for x in range(0, image.shape[1] - ws, stride):
                self.patch_list = np.append(self.patch_list, [[x, y, x+ws, y+ws]], axis=0)
                yield image[y:y + ws, x:x + ws]
    

    def pipe(self, batch_size : int = None, cache_file : str = None, repeat : bool = True):
        """
        Create pipeline for Tensorflow Dataset useful for batching caching and repeat
        """
        dataset = Dataset.from_generator(lambda: self.generator(), 
                                    output_types=(float32, float32),
                                    output_shapes=((self.imshape, ())))
        # Caching if requested
        if cache_file:
            print("Caching the dataset")
            dataset = dataset.cache(cache_file)
        # Indefinitely repeat dataset to avoid errors during training
        if repeat:
            dataset = dataset.repeat()
        # Create batches
        if batch_size:
            dataset = dataset.batch(batch_size)
        # Prefetch
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def regression_pipe(self, batch_size : int = None, cache_file : str = None, repeat : bool = True):
        """
        Create pipeline for Tensorflow Dataset useful for batching caching and repeat
        """
        dataset = Dataset.from_generator(lambda: self.regression_generator(), 
                output_types=(float32, {'classifier_output' : float32, 'regression_output' : float32}),
                output_shapes=((self.imshape, {'classifier_output' : (), 'regression_output' : ()})))
        # Caching if requested
        if cache_file:
            print("Caching the dataset")
            dataset = dataset.cache(cache_file)
        # Indefinitely repeat dataset to avoid errors during training
        if repeat:
            dataset = dataset.repeat()
        # Create batches
        if batch_size:
            dataset = dataset.batch(batch_size)
        # Prefetch
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def split_image_pipe(self, fname : str, ws : int, stride : int):
        dataset = Dataset.from_generator(lambda: self.split_image_generator(fname, ws, stride), 
                                    output_types=(float32),
                                    output_shapes=(ws,ws,3))
        dataset = dataset.map(lambda data: expand_dims(data, 0))

        return dataset
        


if __name__ == '__main__':

    folder = os.path.join('banana-detection', 'bananas_test')
    header = pd.read_csv(os.path.join(folder, "label.csv"), index_col='img_name')
    dataset = BananaDataGenerator(header, os.path.join(folder, "images"))

    patch_gen = dataset.split_image_generator(dataset.fnames[2], ws = 70, stride = 35)

    count = 0
    for _ in range(3):
        plt.figure()
        plt.imshow(next(iter(patch_gen)))
        print("before: \n", dataset.patch_list[count])
        count +=1
    
    mask = np.zeros(shape=(len(dataset.patch_list)), dtype = bool)
    mask[2] = True

    print(dataset.patch_list[mask], dataset.patch_list, mask)

    #train_dataset = tf_dataset_pipe(train_gen, batch_size=20)
    #train_dataset = dataset.pipe()

    #print(next(iter(train_dataset)))