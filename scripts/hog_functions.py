############################################

import torch
from torchvision.ops import nms
from copy import deepcopy
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.transform import rotate
import glob
import numpy as np
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import os
from pathlib import Path

############################################

def get_list_of_images_from_dirs(list_of_dirs):
    '''
    A function to get a single list of files from multiple directories or single directory.
    Should be list of strings or string
    '''

    # if single directory is given as a string, and not a list, convert to list
    if not isinstance(list_of_dirs, list):
        list_of_dirs = [list_of_dirs]

    image_list = []

    # loop over list of directories
    for dir in list_of_dirs:
        images = glob.glob(f'{dir}*.JPG')
        image_list.extend(images)

    return image_list

############################################

def save_list_of_images(images, outdir, basename=None):
    '''
    A function to save out a list of images with sequential numbering.
    If a basename is not specified, image will be used
    '''

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if basename is None:
        for i, image in zip(range(len(images)), images):
            cv2.imwrite(f'{outdir}/image_{i+1}.JPG', image)
    elif isinstance(basename, str):
        for i, image in zip(range(len(images)), images):
            cv2.imwrite(f'{outdir}/{basename}_{i+1}.JPG', image)
    elif isinstance(basename, list):
        for i, image, base in zip(range(len(images)), images, basename):
            cv2.imwrite(f'{outdir}/{base}_{i+1}.JPG', image)
    else:
        print('Please provide correct format for basename')

############################################

def create_hog_features_and_labels(pos_img_list, neg_img_list, hog_parameters, normalize=False):
    '''
    A function to extract hog features and correspoinding labels for a list of images.
    Labels: Neg = 0, Pos = 1
    normalize: Whether or not to apply the sqrt transformation to images before calculating
    hog featire descriptors to reduce illumination effects.
    '''
    # create empty lists to hold HOG feature descriptors
    hog_features = []
    hog_labels = []

    # define HOG parameters
    orientations = hog_parameters['orientations']
    pixels_per_cell = hog_parameters['pixels_per_cell']
    cells_per_block = hog_parameters['cells_per_block']

    # compute HOG features and label them:
    # positive images
    for file in pos_img_list: #this loop enables reading the files in the pos_im_listing variable one by one
        img = cv2.imread(f'{file}') # open the file
        #img = img.resize((64,128))
        # calculate HOG for positive features
        fd = hog(img, orientations, pixels_per_cell, cells_per_block, transform_sqrt=normalize, channel_axis=-1, block_norm='L2')
        hog_features.append(fd)
        hog_labels.append(1)
        
    # negative images
    for file in neg_img_list:
        img = cv2.imread(f'{file}') # open the file
        #img = img.resize((64,128))
        # calculate HOG for negative features
        fd = hog(img, orientations, pixels_per_cell, cells_per_block, transform_sqrt=normalize, channel_axis=-1, block_norm='L2')
        hog_features.append(fd)
        hog_labels.append(0)

    # apply StandardScalar()?
        

    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    hog_labels = le.fit_transform(hog_labels)

    return hog_features, hog_labels

############################################

def svm_model_from_hog_fd(hog_features, hog_labels, test_size = 0.2):
    '''
    A functino that constructs and trains an SVM
    Returns trained SVM
    
    test_size: float between [0, 1] to specify proportion of data to set aside for testing
    '''
    # constrcut linear SCV model
    model = LinearSVC()
    # Apply probability calibration to convert classifier confidence score to probabilities via Platt scaling
    model = CalibratedClassifierCV(model)

    # partition data into training and testing
    train_data, test_data, train_labels, test_labels = train_test_split(
	    np.array(hog_features), hog_labels, test_size=test_size, random_state=42)

    # train SVM
    print(" Training Linear SVM classifier...")
    model.fit(train_data, train_labels)

    # Evaluate the classifier
    print(" Evaluating classifier on test data ...")

    accuracy = model.score(test_data, test_labels)
    print('Accuracy: ', accuracy)

    predictions, _ = svm_model_predict_on_hog_fd(model, test_data)
    print(classification_report(test_labels, predictions))

    return model, model.score(test_data, test_labels), classification_report(test_labels, predictions)

############################################

def svm_model_predict_on_hog_fd(model, hog_features):
    '''
    A function to apply svm classifier to hog feature descriptors.
    Returns:
        predictions: list of class labels
        probs: list of lists length two with probabilities predictions belong to each class
    '''

    predictions = model.predict(hog_features)
    probs = model.predict_proba(hog_features)

    return predictions, probs


############################################

def extract_hog_fd_sliding_window(image, window_size, window_step, hog_parameters, normalize=False):
    '''
    A function that passes a sliding window over image and creates list of hog features, list of hog images,
    and a list of top left corners of each window.

    '''

    image_height = image.shape[0]
    image_width = image.shape[1]

    # define HOG parameters
    orientations = hog_parameters['orientations']
    pixels_per_cell = hog_parameters['pixels_per_cell']
    cells_per_block = hog_parameters['cells_per_block']

    # create empty list to store hog feature descriptors and images
    hog_features = []
    hog_images = []
    label_points = []

    print('Extracting hog feature descriptors...')

    # loop over top left corners defined by window_size in steps of window_step
    for y in range(0, image_height-window_size, window_step):
        for x in range(0, image_width-window_size, window_step):
            window = image[y:y+window_size, x:x+window_size]
            fd, hog_image = hog(window, orientations, pixels_per_cell,
                                            cells_per_block, channel_axis=-1, block_norm='L2', visualize=True, transform_sqrt=normalize)
            hog_features.append(fd)
            hog_images.append(hog_image)
            label_points.append([x, y])

    return hog_features, hog_images, label_points

############################################

def get_positive_preds_and_labels(label_points, predictions, probabilities):
    '''
    A function to get positive predictions, their probabilities and locations
    from outputs of extract_hog_fd_sliding_window and svm_model_predict_on_hog_fd
    '''

    positive_points = [point for (pred, point) in zip(predictions, label_points) if pred == 1]
    positive_probs = [prob[1] for (pred, prob) in zip(predictions, probabilities) if pred == 1]

    return positive_points, positive_probs

############################################

def do_nms(points, probabilities, box_size, iou_threshold=0.2):
    '''
    A function to perform non-max suppression on positive predictions.

    Returns torch tensors of bboxes and their classifier probabilities
    in order of descending proability.

    box_size should be form (x, y) where x is the vertical axis
    '''

    # create list containing bounding boxes corner values to get
    # bbox corners in form: [190, 380, (190+300), (380+150)] required by torch nms
    boxes = []

    for point in points:
        p = deepcopy(point)
        convert_point_to_bbox(p, box_size=box_size)
        boxes.append(p)

    # convert boxes and probabilities to torch tensors
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(probabilities, dtype=torch.float32) # torch.tensor([[p] for p in probs_pos], dtype=torch.float32)

    # implement nms. Returns indices of boxes chosen by nms in boxes
    boxes_nms_idxs = nms(boxes = boxes, scores = scores, iou_threshold=iou_threshold)
    # Filter boxes and probability tensors to contain only nms selected boxes
    boxes_nms = torch.index_select(boxes, 0, boxes_nms_idxs).int() # convert to integer values
    probs_nms = torch.index_select(torch.tensor(probabilities), 0, boxes_nms_idxs)

    return boxes, scores, boxes_nms, probs_nms
    
############################################

def convert_point_to_bbox(point, box_size):
    '''
    A function to convert a point to a list containing bbox parameters: [x0, y0, x0+h, x0+w]

    box_size should be a tuple or list of length two of form (x, y) = (h, w)
    '''

    point.append(point[0]+box_size[0])
    point.append(point[1]+box_size[1])
    
    return point

############################################

def get_prob_quartiles(probabilities):
    '''
    A function to get quartile values of proability distribution in a torch tensor of values.

    Returns an array of form (0, Q25, Q50, Q75, 100) where (0, 100) are the min and max values
    in the tensor.
    '''
    q = torch.tensor([0, 0.25, 0.5, 0.75, 1], dtype=probabilities.dtype)
    return torch.quantile(probabilities, q)
    #return np.linspace(torch.min(probabilities), torch.max(probabilities), 5)

############################################

def get_quartile_cmap(quartiles, tensor, color_map='viridis'):
    '''
    A function to create colour map for a tensor from quartile values of that tensor
    such that each quartile is assigned a different colour.

    Returns a ListedColormap the same length as as tensor.
    '''

    colour_map = cm.get_cmap(color_map, 4)

    # create empty list to store colour values
    colours = []

    for p in tensor:
        if p >= quartiles[3]:
            colours.append(colour_map.colors[0])
        elif p < quartiles[3] and p >= quartiles[2]:
            colours.append(colour_map.colors[1])
        elif p < quartiles[2] and p >= quartiles[1]:
            colours.append(colour_map.colors[2])
        else:
            colours.append(colour_map.colors[3])

    return ListedColormap(colours)

############################################

def plot_image(ax, image, boxes=None, display_bounds=None, colour_map=None, title='Image', idx_labels=False):
    '''
    A function that plots the original image, the model predictions before nms, and the model predictions after nms.
    The predicted bounding boxes are divided into quartiles with confidence of prediction.

    display_bounds: list [xmin, ymin, xmax, ymax] to crop the display to. If None, displays whole image
    '''

    if display_bounds is None:
        xmin, xmax, ymin, ymax = 0, image.shape[0], 0, image.shape[1]
    else:
        xmin, xmax, ymin, ymax = display_bounds[0], display_bounds[1], display_bounds[2], display_bounds[3]

    #ax.axis('off')
    ax.imshow(image[xmin:xmax, ymin:ymax])
    ax.set_title(title, fontsize=18)

    if boxes is not None:
        # if colour map isn't specified, create listed colour map of same colour the same length as tesnor boxes
        if colour_map is None:
            colour_map = ListedColormap(['red' for box in boxes])
        for i, box in enumerate(boxes):
            if box[0] > ymin and box[1] > xmin and box[2] < ymax and box[3] < xmax:
                ax.add_patch(Rectangle(((box[0]-ymin, box[1]-xmin)), (box[2]-box[0]), (box[3]-box[1]),
                            edgecolor=colour_map.colors[i],
                            facecolor='none',
                            lw=1))
                if idx_labels is True:
                    ax.text((box[0]-ymin), (box[1]-xmin+24),
                                s = f'{i}',
                                color='white',
                                fontsize=12) 

############################################

def augment_with_zero_choices():
    '''
    A function to augment dataset and train svm using only augmentation.
    Original data will be randomly flipped or not, and then randomly rotated, or not.
    '''

    ############################################