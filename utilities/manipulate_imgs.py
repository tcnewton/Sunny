import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import cv2

##### mean - std image

def mean_std_img(img_train):
    # input:
    # img_train : [batches,height,width,1]
    # dimshape : tuple (height,width)
    #output:
    # img_train: img mean - std normalizate, mean, std
    dimshape = tuple(img_train.shape[1:3])
    scaler = StandardScaler()
    img_train = np.squeeze(img_train,3)
    img_train = np.reshape(img_train,(img_train.shape[0],-1))
    img_train = scaler.fit_transform(img_train)
    img_train = np.reshape(img_train,[img_train.shape[0],dimshape[0],dimshape[1],1])
    return img_train,scaler.mean_,scaler.scale_

def mean_std(img_train,dimshape):
    # input:
    # img_train : [batches,height,width,1]
    # dimshape : tuple (height,width)
    #output:
    # img_train: img mean - std normalizate, mean, std
    scaler = StandardScaler()
    img_train = np.squeeze(img_train,3)
    img_train = np.reshape(img_train,(img_train.shape[0],-1))
    img_train = scaler.fit_transform(img_train)
    img_train = np.reshape(img_train,[img_train.shape[0],dimshape[0],dimshape[1],1])
    return img_train,scaler.mean_,scaler.scale_

def OHE_lbl(ylabel):
    encoder = OneHotEncoder(categories='auto')
    #encoder = OneHotEncoder() # python version 2
    sample,height,width = ylabel.shape[:3]
    # fazendo o one-hot-encoded
    y_valid_ohe = encoder.fit_transform(ylabel.reshape(-1, 1))
    y_valid_ohe = y_valid_ohe.toarray()
    y_valid_ohe = np.reshape(y_valid_ohe, [sample, height, width, -1])
    return y_valid_ohe

def hist_equalize(img):
    if img.ndim >=4:
        img = np.squeeze(img,3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for ii in range(img.shape[0]):
        cl1 = clahe.apply(img[ii].astype(np.uint8))
        img[ii] = cl1
    return img

def gera_bb_maskbin(mask):
    '''input imagem binaria do tipo array (0-1) gera
    output array (xmin,ymin,width,height)
    e.g.
    bb_mask = gera_bb_maskbin(mask)'''
    linha = np.where(np.any(mask,axis=1))
    coluna = np.where(np.any(mask,axis=0))
    xmin = np.min(coluna)
    xmax = np.max(coluna)
    ymin = np.min(linha)
    ymax = np.max(linha)
    bb_teste = np.array([xmin,ymin,xmax,ymax])
    return bb_teste


# mudar a probabilidade do data_augm para .3
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-5, 5), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_REFLECT_101, prob=0.3, prob_teste=False):
    '''input: image: [height,width],mask :[height,width]
    outputs prob_teste: boolean, image[height,width],mask[height,width]'''
    if np.random.random() < prob:
        prob_teste = True
        height, width, = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode)
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode)

    return prob_teste, image, mask

def make_crop_img(img,mask,img_size):
    assert img.shape == mask.shape,"image and mask has to b same shape"
    if img.ndim >2:
        img = np.squeeze(img,3)
        mask = np.squeeze(mask,3)
    bb_mask = []
    for ii in range(len(img)):
        bb = gera_bb_maskbin(mask[ii])
        cx = (bb[0]+bb[2])//2
        cy = (bb[1]+bb[3])//2
        width = bb[2]-bb[0]+1
        height = bb[3]-bb[1]+1
        dim_bb = np.max([width,height]) # dim square
        dim_bb = dim_bb//2 + 10
        sex = np.max([cx-dim_bb,0])
        sey = np.max([cy-dim_bb,0])
        idx = np.min([cx+dim_bb,img_size-1])
        idy = np.min([cy+dim_bb,img_size-1])
        img_crop = np.zeros_like(img[ii])
        img_crop[sey:idy,sex:idx] = img[ii,sey:idy,sex:idx]
        bb_mask.append(img_crop)
    bb_mask = np.array(bb_mask)
    bb_mask = np.expand_dims(bb_mask,3)
    return bb_mask

def make_crop_bb_yolo(img,mask,img_size,bb_array):
    assert img.shape == mask.shape, "image and mask has to b same shape"
    if img.ndim > 2:
        img = np.squeeze(img, 3)
    bb_mask = []
    mask_strat = []
    for ii in range(len(img)):
        bb = bb_array[ii]
        if np.max(bb)>0:
            cx = (bb[0] + bb[2]) // 2
            cy = (bb[1] + bb[3]) // 2
            width = bb[2] - bb[0] + 1
            height = bb[3] - bb[1] + 1
            dim_bb = np.max([width, height])  # dim square
            dim_bb = dim_bb // 2 + 10
            sex = np.max([cx - dim_bb, 0])
            sey = np.max([cy - dim_bb, 0])
            idx = np.min([cx + dim_bb, img_size - 1])
            idy = np.min([cy + dim_bb, img_size - 1])
            img_crop = np.zeros_like(img[ii])
            img_crop[sey:idy, sex:idx] = img[ii, sey:idy, sex:idx]
            bb_mask.append(img_crop)
            mask_strat.append(mask[ii])
    bb_mask = np.array(bb_mask)
    bb_mask = np.expand_dims(bb_mask, 3)
    mask_strat = np.array(mask_strat)
    return bb_mask , mask_strat
