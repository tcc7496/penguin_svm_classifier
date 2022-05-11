'''
A script to augment dataset by chosen method.
'''

import click
from data_manipulation_fns import augment_seq_process, augment_each_image_sep

############################################

@click.command()
@click.argument('image_dir', type=click.Path(exists=True))
@click.option('-o', '--outputdir', type=click.Path(), help='Output directory to store augmented images. If not defined,')
@click.option('-a', '--angles', type=list, help='')
@click.option('-m', '--method', type=click.Choice(['sequential', 'separate']), case_sensitive=False)
@click.option('--plot_sample', is_flag=True, default=False)
def main(image_dir, outputdir, angles, method, plot_sample):
    '''
    '''
    if plot_sample:
        show_sample = True
    else:
        show_sample = False

    if method == 'sequential':
        augment_seq_process(image_dir=image_dir, outputdir=outputdir, angles=angles, show_sample=show_sample)
    elif method == 'separate':
        augment_each_image_sep(image_dir=image_dir, outputdir=outputdir, angles=angles, show_sample=show_sample)
    else:
        click.echo('Please choose a method to augment dataset')

############################################
     
if __name__ == "__main__":
    '''
    '''
    main() 
