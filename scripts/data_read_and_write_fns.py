############################################

import os
import cv2
import glob

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