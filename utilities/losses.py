import sklearn.metrics as metrics
import numpy as np

def calc_metric(y_true,y_pred):
    '''calculate dice from 2 samples of images
    inputs:
    y_true: [sample,height,width,OHE].flatten()
    y_pred: [sample,height,width,OHE].flatten()
    outputs:
    dice distance'''
    #y_true = y_true.flatten()
    #y_pred = y_pred.flatten()
    f1_zero = metrics.f1_score(y_true, y_pred)
    jacc_x = metrics.jaccard_similarity_score(y_true, y_pred)
    return [f1_zero,jacc_x]

def dice_loss(y_gt,x_pred):
    assert y_gt.ndim == x_pred.ndim,"ground truth and predict must have same shape"
    x = int(y_gt.shape[-1])
    background = 0
    interno = 1
    externo = 2
    intersection = np.multiply(y_gt,x_pred)
    while y_gt.ndim>1:
        x_pred = np.sum(x_pred,axis=0)
        y_gt = np.sum(y_gt,axis=0)
        intersection = np.sum(intersection,axis=0)
    dice_background = 2* intersection[background] / (x_pred[background]+y_gt[background])
    dice_interno = 2* intersection[interno] / (x_pred[interno]+y_gt[interno])
    if x>2:
        dice_externo = 2* intersection[externo] / (x_pred[externo]+y_gt[externo])
        dice = tuple([dice_background,dice_interno,dice_externo])
    else:
        dice = tuple([dice_background,dice_interno])
    return dice
