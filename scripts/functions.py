import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    '''
    '''