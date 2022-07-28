import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import click
import joblib
import os
from pathlib import Path
import ast
import pandas as pd
import numpy as np
from hog_functions import extract_hog_fd_sliding_window, svm_model_predict_on_hog_fd, get_positive_preds_and_labels, do_nms, get_prob_quartiles, get_quartile_cmap, plot_image

############################################


@click.command()
@click.argument('image', type=click.Path(exists=True))
@click.argument('model', type=click.Path(exists=True))
@click.argument('hog_parameters', type=click.Path(exists=True))
@click.argument('outputdir', type=click.Path())
@click.option('-sz', '--window_size', type=int, default=24, help='Window size over which to calculate hog in image.')
@click.option('-st', '--window_step', type=int, default=6, help='Step size to move window oer image between hog calculations.')
@click.option('-iou', '--iou_threshold', type=float, default=0.2, help='Intersection over union threshold to use when performing non-max suppression.')
@click.option('-p', '--prob_threshold', type=float, default=0.5, help='Probability threshold above which to keep positive identifications. If not specified, everything above 0.5 is kept, i.e. more likely to be penguin than background.')
@click.option('-q', '--quartile_cut', type=click.IntRange(min=1, max=3), default=None, help='If specified, positive indentifications will filtered based on quartie values. Value should be number of quartiles to cut from bottom. For example, -q "1" will filter out the bottom quarter of data. Overides --prob_threshold if specified.')
def main(image, model, hog_parameters, outputdir, window_size, window_step, iou_threshold, prob_threshold, quartile_cut):
    '''
    '''
    # read in image
    img_pred = cv2.imread(image)

    print(img_pred.shape)

    # read hog parameters from text file
    with open(hog_parameters, 'r') as f:
        dict = f.read()
    # convert from str type to dictionary
    hog_parameters = ast.literal_eval(dict)

    # calculate hog features
    hog_features, hog_images, label_points = extract_hog_fd_sliding_window(img_pred,
                                                                           window_size=window_size, window_step=window_step,
                                                                           hog_parameters=hog_parameters)
    print('hog features calculated')
    print(len(hog_features), len(label_points))

    # load model
    m = joblib.load(model)

    # make predictions on hog features of new image
    preds, probs = svm_model_predict_on_hog_fd(
        model=m, hog_features=hog_features)

    # deep copy labels and probabilities
    label_points_cp = deepcopy(label_points)
    probs_cp = deepcopy(probs)

    # extract positive predictions and their locations
    positive_points, positive_probs = get_positive_preds_and_labels(
        label_points=label_points_cp, predictions=preds, probabilities=probs_cp)

    # do nms
    boxes, scores, boxes_nms, probs_nms = do_nms(points=positive_points, probabilities=positive_probs,
                                                 box_size=(window_size, window_size), iou_threshold=iou_threshold)

    ## Outputs ##

    # create output directory if it doesn't exist
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    # get image and model names to output and convert to string
    image_name = str(Path(image).stem)
    model_name = str(Path(model).stem)

    # convert boxes left after nms to dataframe for saving out
    boxes_nms_df = pd.DataFrame(boxes_nms).astype('int')
    boxes_nms_df.columns = ['ymin', 'xmin', 'ymax', 'xmax']
    boxes_nms_df['prob'] = probs_nms.numpy()
    # boxes df already sorted by probability with highest prob first

    # get quartiles for colour mapping and for filtering bboxes by prob if needed
    quartiles = get_prob_quartiles(probs_nms)
    quartiles_np = quartiles.numpy()  # convert to numpy array
    print('quartiles:', quartiles)

    # filter data based on probability if specified
    if quartile_cut is not None:
        # print number of rows in df for checks
        print('Number of bboxes before filtering:', boxes_nms_df.shape[0])
        # get probability threshold from quartile list calculated above.
        # round to float16 to provide more definite cut-off
        prob_threshold = np.float16(quartiles_np[quartile_cut])

    # Filter bboxes based on prob_threshold. If neither quartile_cut or prob_threshold is specified in command line,
    # all data should be kept
    print('Probability threshold:', prob_threshold)
    boxes_nms_df = boxes_nms_df.loc[boxes_nms_df['prob'] > prob_threshold]
    print('Number of bboxes after filtering:', boxes_nms_df.shape[0])

    # save bboxes as csv
    boxes_nms_df.to_csv(
        f'{outputdir}/{image_name}_{model_name}_bboxes.csv', sep=" ", header=True)

    # save out quartiles
    with open(f'{outputdir}/{image_name}_{model_name}_quartiles.txt', 'w') as data:
        data.write(str(quartiles_np))

    # make colour map
    # cm_quartiles = get_quartile_cmap(quartiles=quartiles, tensor=probs_nms)
    cm_quartiles = get_quartile_cmap(
        quartiles=quartiles, tensor=boxes_nms_df['prob'])

    ### make plot ###

    # define sub-region if wanted
    xmin = 0
    xmax = img_pred.shape[0]
    ymin = 0
    ymax = img_pred.shape[1]

    # create list of sub-region bounds to feed into plot functions
    display_bounds = [xmin, xmax, ymin, ymax]

    fig, ax1 = plt.subplots(figsize=(30, 30))

    plot_image(ax=ax1, image=img_pred, boxes=boxes_nms_df, title='Prediction',
               colour_map=cm_quartiles, display_bounds=display_bounds, idx_labels=True)

    plt.savefig(f'{outputdir}/{image_name}_{model_name}_bboxes.jpg', dpi=300)

############################################


if __name__ == "__main__":
    '''
    '''
    main()
