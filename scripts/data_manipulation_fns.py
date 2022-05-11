############################################

import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path
from data_read_and_write_fns import get_list_of_images_from_dirs, save_list_of_images

############################################

def augment_seq_process(image_dir, outputdir=None, angles=[0, 90, 180, 270], show_sample=False):
    '''
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
    orig_imgs = np.array([np.asarray(cv2.imread(img)) for img in img_path_list])

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

############################################

def augment_each_image_sep(image_dir, outputdir=None, angles=[90, 180, 270], show_sample=False):
    '''
    A function that takes a directory or images and randomly decides whether to rotate or flip them,
    and then randomly assigns the flip or rotation.
    '''

    # set random number generator seed
    np.random.seed(1)

    # get list of images
    img_path_list = get_list_of_images_from_dirs(image_dir)

    # make array of images
    orig_imgs = np.array([np.asarray(cv2.imread(img)) for img in img_path_list])

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
            augment = rotate(img, angle=np.random.choice(angles), mode='wrap')
            augmented.append(augment)
            #ax2.imshow(augment)
        
        #plt.show()

    # create output directory if none given
    if outputdir is None:
        outputdir = f'{image_dir}/augmented_each_img_sep/'

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # cretae list of image names to use as basenames in save_list_of_images
    img_name_list = [str(Path(image).stem) for image in img_path_list]
    
    # save images out
    save_list_of_images(augmented, outputdir, basename=img_name_list)

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

if __name__ == "__main__":
    '''
    '''