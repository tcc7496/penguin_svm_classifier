import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import click
import joblib
import os
from pathlib import Path
#from pathlib import PurePath
import ast
import pandas as pd
from hog_functions import extract_hog_fd_sliding_window, svm_model_predict_on_hog_fd, get_positive_preds_and_labels, do_nms, get_prob_quartiles, get_quartile_cmap, plot_image

@click.command()
@click.argument('image', type=click.Path(exists=True))
@click.argument('model', type=click.Path(exists=True))
@click.argument('hog_parameters', type=click.Path(exists=True))
@click.argument('outputdir', type=click.Path())
@click.option('-sz', '--window_size', type=int, default=24, help='')
@click.option('-st', '--window_step', type=int, default=6, help='')
@click.option('-iou', '--iou_threshold', type=float, default=0.2, help='Intersection over union threshold to use when performing non-max suppression.')
def main(image, model, hog_parameters, outputdir, window_size, window_step, iou_threshold):
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
    preds, probs = svm_model_predict_on_hog_fd(model = m, hog_features = hog_features)

    # deep copy labels and probabilities
    label_points_cp = deepcopy(label_points)
    probs_cp = deepcopy(probs)

    # extract positive predictions and their locations
    positive_points, positive_probs = get_positive_preds_and_labels(label_points=label_points_cp, predictions = preds, probabilities=probs_cp)

    # do nms
    boxes, scores, boxes_nms, probs_nms = do_nms(points = positive_points, probabilities=positive_probs,
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
    
    # save bboxes as csv
    boxes_nms_df.to_csv(f'{outputdir}/{image_name}_{model_name}_bboxes.csv', sep = " ", header = True)

    # get quartiles for colour mapping
    quartiles = get_prob_quartiles(probs_nms)
    print('quartiles:', quartiles)

    # save out quartiles
    quartiles_np = quartiles.numpy()
    with open(f'{outputdir}/{image_name}_{model_name}_quartiles.txt', 'w') as data: 
      data.write(str(quartiles_np))

    # make colour map
    cm_quartiles = get_quartile_cmap(quartiles=quartiles, tensor=probs_nms)

    ### make plot ###

    # define sub-region if wanted
    xmin = 0
    xmax = img_pred.shape[0]
    ymin = 0
    ymax = img_pred.shape[1]

    # create scaling factor for figure height depending on sub-region
    fig_height = 40000/(ymax-ymin)

    # create list of sub-region bounds to feed into plot functions
    display_bounds = [xmin, xmax, ymin, ymax]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, fig_height), sharex=True, sharey=True)

    plot_image(ax=ax1, image=img_pred, title='Prediction Image', display_bounds=display_bounds)
    plot_image(ax=ax2, image=img_pred, boxes=boxes, title='Before NMS', display_bounds=display_bounds)
    plot_image(ax=ax3, image=img_pred, boxes=boxes_nms, title='After NMS', colour_map=cm_quartiles, display_bounds=display_bounds)

    plt.savefig(f'{outputdir}/{image_name}_{model_name}_bboxes.jpg', dpi=300)


if __name__ == "__main__":
    '''
    '''
    main()                      