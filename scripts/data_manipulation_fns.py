############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import rotate
import os
import cv2
from pathlib import Path
from data_read_and_write_fns import get_list_of_images_from_dirs, save_list_of_images

############################################

def augment_seq_process(image_dir, outputdir=None, angles=[0, 90, 180, 270], show_sample=False, numbering=False):
    '''
    !!!! The rotate() in this function still needs to be switched over to cv2 method instead of skimage !!!!

    A function to augment dataset and train svm using only augmentated images.
    Original data will be randomly flipped or not, and then randomly rotated.
    The augmentation is done sequentially such that the output of the flipping
    is the input to the rotation.
    '''
    # set random number generator seed
    np.random.seed(1)

    # get list of images
    img_path_list = get_list_of_images_from_dirs(image_dir)
    
    # make array of images
    # think this is causing issues when the second dimensions don't match. Only needed for plotting so comment out atm.
    # orig_imgs = np.array([np.asarray(cv2.imread(img)) for img in img_path_list])

    # decide randomly whether to flip images and which flip to do
    flipped = [cv2.flip(img, np.random.choice([0, 1, -1])) if np.random.choice([0, 1]) else img for img in orig_imgs]

    # decide randomly whether to rotate and which rotation to do
    rotated = [rotate(img, angle=np.random.choice(angles), mode='wrap') for img in flipped]

    # create output directory if none given
    if outputdir is None:
        outputdir = f'{image_dir}/augmented_seq_process/'

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # cretae list of image names to use as basenames in save_list_of_images
    img_name_list = [str(Path(image).stem) for image in img_path_list]
    
    # save images out
    save_list_of_images(rotated, outputdir, basename=img_name_list)

    """
    Commenting out for now because need orig_imgs object which is causing issues.
    if show_sample is True:
        plt.figure(figsize=(9,9))
        i = 0
        for img in orig_imgs[0:16]:
            plt.subplot(4, 4, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img)
            i += 1
        plt.suptitle("Sample after augmentation", fontsize=20)
        plt.show()
    """

############################################

def augment_each_image_sep(image_dir, outputdir=None, angles=[90, 180, 270], show_sample=False, numbering=False):
    '''
    A function that takes a directory or images and randomly decides whether to rotate or flip them,
    and then randomly assigns the flip or rotation.
    '''

    # set random number generator seed
    np.random.seed(1)

    # get list of images
    img_path_list = get_list_of_images_from_dirs(image_dir)
    print('No. of images to augment:', len(img_path_list))

    # make array of images
    # think this is causing issues when the second dimensions don't match. Only needed for plotting so comment out atm.
    # orig_imgs = np.array([np.asarray(cv2.imread(img), dtype=object) for img in img_path_list])
    

    augmented = []
    # loop over image files
    for file in img_path_list:
        img = cv2.imread(file) # open image
        # plot original image and augmented one next to each other for testing
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
        #ax1.imshow(img)
        # choose randomly whether to flip or rotate image
        j = np.random.choice([0, 1])

        # flip image if j=0
        if j == 0:
            augment = cv2.flip(img, np.random.choice([0, 1, -1]))
            augmented.append(augment)
            #ax2.imshow(augment)
        # rotate image if j=1
        if j == 1:
            angle = np.random.choice(angles)
            if angle == 90:
                augment = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                augment = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                augment = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                print('Please choose an angle from [90, 180, 270].')
            #augment = rotate(img, angle=np.random.choice(angles), mode='wrap')
            augmented.append(augment)
            #ax2.imshow(augment)
        
        #plt.show()
    
    # create output directory if none given
    if outputdir is None:
        outputdir = f'{image_dir}/augmented_each_img_sep/'

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # create list of image names to use as basenames in save_list_of_images
    img_name_list = [f'{str(Path(image).stem)}_aug' for image in img_path_list]
    
    # save images out
    save_list_of_images(augmented, outputdir, basename=img_name_list, numbering=numbering)

    """
    Commenting out for now because need orig_imgs object which is causing issues.
    if show_sample is True:
        plt.figure(figsize=(9,9))
        i = 0
        for img in orig_imgs[0:16]:
            plt.subplot(4, 4, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img)
            i += 1
        plt.suptitle("Sample after augmentation", fontsize=20)
        plt.show()
    """

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

def get_bbox_from_centroid_w_h(xc, yc, h, w):
    """
    xc, yc = co-orindates of centroid
    w, h = width and height of bbox

    in pixels
    
    """

    x0 = xc - np.floor(0.5*h) # floor values incase of 0.5 after dividing by 2
    y0 = yc - np.floor(0.5*w)
    x1 = xc + np.ceil(0.5*h)
    y1 = yc + np.ceil(0.5*w)

    return x0, y0, x1, y1

############################################

def read_yolo_labels(file):
    """
    A function to read a yolov5 labels text file into a pandas dataframe

    """

    return pd.read_csv(file, sep = " ", names = ["class_no", "yc", "xc", "w", "h"], header = None)

############################################

def unstandardise_yolo_labels(df, x_max, y_max):
    """
    A function to convert yolo label coorindates from (0,1] interval to pixels.
    Assumes yolo labels are labelled as below.
    x_max, y_max should be in pixels
    """
    df['yc'] = round(df['yc']*y_max)
    df['xc'] = round(df['xc']*x_max)
    df['w'] = round(df['w']*y_max)
    df['h'] = round(df['h']*x_max)

    return df

############################################

def get_thumbnails_from_image_labels(image, labels, definite=True):
    '''
    A function that saves out thumbnail images from a larger image using
    label bounding boxes saved in yolo label format.
    
    image: image file with labelled objects
    labels: yolo label format file
    definite: If True, will filter for labels only in class 0 which correspond to a definite adult king penguin

    '''

    # read in image
    img = cv2.imread(image)
    basename = os.path.splitext(os.path.basename(image))[0]

    # get max number of pixels
    x_max, y_max, _ = img.shape

    # read in labels
    labels = read_yolo_labels(labels)
    # convert yolo label format to pixels
    labels_px = unstandardise_yolo_labels(df = labels, x_max = x_max, y_max = y_max)

    if definite:
        # filter labels for only definite penguin class
        labels_px = labels_px[labels_px['class_no'] == 0]

    # find bounding box coordinates
    x0, y0, x1, y1 = get_bbox_from_centroid_w_h(xc = labels_px['xc'], yc = labels_px['yc'], h = labels_px['h'], w = labels_px['w'])

    # get lists for x0, y0, x1, y1
    x0 = [int(item) for item in list(x0)]
    y0 = [int(item) for item in list(y0)]
    x1 = [int(item) for item in list(x1)]
    y1 = [int(item) for item in list(y1)]

    # create empty list to store sample images
    samples = []

    # loops over lists of centroids
    for x0, y0, x1, y1 in zip(x0, y0, x1, y1):
        sample = img[x0:x1, y0:y1]
        samples.append(sample)

    return samples, basename

############################################

def get_thumbnails_around_centroid_from_image_labels(image, labels, dims, definite=True, include_edge_cases=True):
    '''
    A function to get thumbnails of standardised size around centroids of labelled objects
    in a larger image.

    image: image file with labelled objects
    labels: yolo label format file
    dims: tuple in form (w, h) where w and h are integers
    definite: If True, will filter for labels only in class 0 which correspond to a definite adult king penguin 
    include_edge_cases: If True will include labelled features at edges which will be smaller than the dims
    specified when trimmed to fit inside the image.
    
    '''

    # read in image
    img = cv2.imread(image)
    basename = os.path.splitext(os.path.basename(image))[0]

    # get max number of pixels
    x_max, y_max, _ = img.shape

    # read in labels
    labels = read_yolo_labels(labels)
    # convert yolo label format to pixels
    labels_px = unstandardise_yolo_labels(df = labels, x_max = x_max, y_max = y_max)

    if definite:
        # filter labels for only definite penguin class
        labels_px = labels_px[labels_px['class_no'] == 0]

        # get lists of centroids and convert to integers
        xc = [int(item) for item in list(labels_px['xc'])]
        yc = [int(item) for item in list(labels_px['yc'])]

        # create empty list to store sample images
        samples = []

        # get width and height
        h, w = int(dims[1]), int(dims[0])
        print('(h, w)', h, w)

        # loops over lists of centroids
        for x, y in zip(xc, yc):
            print('(x, y)', x, y)
            print('(x-h, y-w)', x-h//2, y-w//2)
            
            if (x < h//2 or y < w//2) and include_edge_cases is False:
                print('Skipping this label because we dont want edge cases here.')
                continue

            if x < h//2:
                x = h//2

            if y < w//2:
                y = w//2
  
            
            print('(x, y)', x, y)
            print('next')
            sample = img[(x-h//2):(x+h//2), (y-w//2):(y+w//2)]

            samples.append(sample)

        return samples, basename
    
############################################

if __name__ == "__main__":
    '''
    '''
    img_dir = '/Users/taracunningham/projects/PhD/data/King_Penguin_Surveys/St_Andrews_Bay/210216_00668_200-200/210216_00668_200-200_1-15'

    outdir = '/Users/taracunningham/projects/PhD/data/King_Penguin_Surveys/St_Andrews_Bay/210216_00668_200-200/210216_00668_200-200_1-25to1-25_thumbs_augmented'

    augment_each_image_sep(img_dir, outputdir=outdir, angles=[90, 180, 270], show_sample=False, numbering=False)

