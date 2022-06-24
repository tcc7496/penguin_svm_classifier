'''
A script that takes an image and the corresponding labels file and saves the label
bounding boxes as images into a desired output directory.
'''

import click
from data_manipulation_fns import get_thumbnails_from_image_labels, get_thumbnails_around_centroid_from_image_labels
from data_read_and_write_fns import save_list_of_images

############################################

@click.command()
@click.argument('image', type=click.Path(exists=True))
@click.argument('labels', type=click.Path(exists=True))
@click.argument('outputdir', type=click.Path())
#@click.option('--bboxes', 'mode', flag_value='bboxes', default=True)
#@click.option('--sized', 'mode', flag_value='sized')
@click.option('--mode', type=click.Choice(['bboxes', 'sized']))
@click.option('-dims', nargs=2, type=(int, int), default=(24, 24), help='The dimensions of the bounding boxes to draw in pixels (w, h)')
@click.option('-d', '--definite_object', is_flag=True, default=True, show_default=True, help='Option to filter only for definite adult penguin labels (class 0). Default is True.')
def main(image, labels, outputdir, mode, dims, definite_object):
    '''
    Runs in two modes:
    bboxes: Thumbnails use bounding boxes specified in the label file.
    sized:  Thumbnails are a user-defined size in pixels specified in 'dims'surrounding
            the centroid point specifed in the label file.
    '''

    if definite_object:
        dflag=True
    else:
        dflag=False

    if mode == 'bboxes':
        samples, basename = get_thumbnails_from_image_labels(image, labels, definite=dflag)   
    elif mode == 'sized':
        samples, basename = get_thumbnails_around_centroid_from_image_labels(image, labels, dims=dims, definite=dflag)


    save_list_of_images(images = samples, outdir = outputdir, basename = basename)
############################################

if __name__ == "__main__":
    '''
    '''
    main()                      