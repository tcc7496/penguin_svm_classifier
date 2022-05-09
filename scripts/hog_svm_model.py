'''
A script to calculate hog features on a set of images and train a Linear SVM model.
'''

# imports
import click
import joblib
import os
from pathlib import Path
from hog_functions import get_list_of_images_from_dirs, create_hog_features_and_labels, svm_model_from_hog_fd

@click.command()
@click.argument('indirlistpos', type=click.Path(exists=True))
@click.argument('indirlistneg', type=click.Path(exists=True))
@click.option('-ho', '--hog_orientations', type=int, default=9, help='Number of orientations when creating hog feature descriptors')
@click.option('-hpix', '--hog_pixels_per_cell', nargs=2, type=(int, int), default=(6,6), help='Number of pixels per cell to use when calculating hog feature descriptors.')
@click.option('-hc', '--hog_cells_per_block', nargs=2, type=(int, int), default=(4,4), help='Number of cells per block to use when calculating hog feature descriptors.')
@click.option('-tf', '--test_fraction', type=float, default=0.3, help='Fraction of training data to be saved for model evaluation. Float on interval [0,1)')
@click.option('-o', '--outputdir', default='./new_model', type=click.Path(), help='Output directory and filename for trained svm model.')
def main(indirlistpos, indirlistneg, hog_orientations, hog_pixels_per_cell, hog_cells_per_block, test_fraction, outputdir):
    '''
    '''

    # read in list of files
    pos_img_dirs = open(indirlistpos).read().splitlines()
    neg_img_dirs = open(indirlistneg).read().splitlines()

    # get list of images from directories
    pos_img_list = get_list_of_images_from_dirs(pos_img_dirs)
    neg_img_list = get_list_of_images_from_dirs(neg_img_dirs)

    # print number of images in positive and negative datasets
    print('No. positive images:', len(pos_img_list), ' No. negative images: ', len(neg_img_list))

    # create dictionary of hog parameters
    hog_parameters = {
        'orientations': hog_orientations,
        'pixels_per_cell': hog_pixels_per_cell,
        'cells_per_block': hog_cells_per_block
    }

    hog_features, hog_labels = create_hog_features_and_labels(pos_img_list, neg_img_list, hog_parameters, normalize=True)

    # create model
    model, accuracy, report = svm_model_from_hog_fd(hog_features=hog_features, hog_labels=hog_labels, test_size=test_fraction)

    # create folder to put model and evaluation
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # construct output file path and stem of output file name in a one-er
    outfile = f'{outputdir}/{str(Path(outputdir).stem)}'

    # save model
    joblib.dump(model, f'{outfile}.npy')

    # save model evaluation to text file
    with open(f'{outfile}_eval.txt', 'w') as f:
        print('Accuracy: ', accuracy, file=f)
        print('Classification Report:\n', report, file=f)

    # write hog parameters to text file
    with open(f'{outfile}_hog_params_dict.txt','w') as data: 
      data.write(str(hog_parameters))


if __name__ == "__main__":
    '''
    '''
    main()