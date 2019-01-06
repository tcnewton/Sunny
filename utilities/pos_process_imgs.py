import os, json, cv2
import numpy as np


## leitura dos arquivos json -- output yolo

def read_json(crop_size):
    json_path = os.path.join(os.getcwd(),'yolo',crop_size,'test','out')
    json_list = [x for x in os.listdir(json_path) if x.endswith('.json')]
    x_array=[]
    for each in json_list:
        f = open(os.path.join(json_path,each),'r')
        r = json.loads(f.read())
        if r:
            xmin = r[0]['topleft']['x']
            ymin = r[0]['topleft']['y']
            xmax = r[0]['bottomright']['x']
            ymax = r[0]['bottomright']['y']
        else:
            xmin,ymin,xmax,ymax = [0,0,0,0] # find no bounding box
        x_array.append([xmin,ymin,xmax,ymax])
    x_array = np.array(x_array)
    return x_array


def compute_i(img,BB):
    """Calculates intersection between contour and bounding box, return contour with large intersection.
    img: list [number_of_contours (x1,y1)]
    BoundingBox: (x1,y1,x2,y2)

    output: img with intersection BB
    """
    # Calculate intersection areas
    _, c_img, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area_intersect = 0
    contour = None
    if c_img and sum(BB)!=0:
        for each in c_img:
            x1 = np.maximum(np.min(each[:,:,0]),BB[0])
            x2 = np.minimum(np.max(each[:,:,0]),BB[2])
            y1 = np.maximum(np.min(each[:,:,1]),BB[1])
            y2 = np.minimum(np.max(each[:,:,1]),BB[3])
            intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
            if intersection > area_intersect:
                contour = each
    if area_intersect==0 and c_img:
        lengths = []
        for coord in c_img:
            lengths.append(len(coord))
        contour = c_img[np.argmax(lengths)]
    return contour


def softmax_fcn(*args):
    total_exp_arg = np.sum(np.exp(args))
    exp_arg = np.exp(args)
    soft = exp_arg/total_exp_arg
    ind = np.argmax(soft, axis=None)
    return int(ind)


def squeeze_predict(x_data_pred):
    '''input pred with 3 ohe classes - output pred with range 0-2 equal classes
    x_data_pred :[samples,width,height,ohe_classes]
    output :
    imfinal [samples,width,height,1]'''
    ch0_arr = np.asarray (x_data_pred[:,:,:,0])
    ch1_arr = np.asarray (x_data_pred[:,:,:,1])
    ch2_arr = np.asarray (x_data_pred[:,:,:,2])
    ch_shape = ch0_arr.shape
    # transformando array em um vetor
    ch0_arr = np.reshape(ch0_arr,[ch_shape[0]*ch_shape[1]*ch_shape[2]])
    ch1_arr = np.reshape(ch1_arr,[ch_shape[0]*ch_shape[1]*ch_shape[2]])
    ch2_arr = np.reshape(ch2_arr,[ch_shape[0]*ch_shape[1]*ch_shape[2]])

    im_final = np.zeros(ch0_arr.shape)
    for ii in range(len(ch0_arr)):
        im_final[ii] = softmax_fcn(ch0_arr[ii],ch1_arr[ii],ch2_arr[ii])
    im_final = np.reshape(im_final,ch_shape)
    return im_final