import os
import numpy as np
from utilities.utils import get_SAX_SERIES,center_crop, YOLO_xml,create_name_file
from utilities.net_utils import UNET_SB
from utilities.manipulate_imgs import hist_equalize, OHE_lbl, gera_bb_maskbin, randomShiftScaleRotate,mean_std_img, make_crop_img, make_crop_bb_yolo
from utilities.manipulate_files import shrink_case, Contour, read_contour, read_merge_contour, map_all_contours,export_all_contours, export_merge_all_contours
from utilities.pos_process_imgs import read_json, compute_i,squeeze_predict
from utilities.losses import calc_metric, dice_loss
import cv2
import keras
import argparse

seed = 1234
weights = None
file_path = os.getcwd()
sax = get_SAX_SERIES(file_path)

parser = argparse.ArgumentParser()
parser.add_argument('net',choices=['UNET','UNET_NORM','FCN'])
parser.add_argument('--model',dest='filename',default='E:/Talles/NN/datasets/sunnybrook')
parser.add_argument('-s', '--size', type=int, dest='crop_size',default=128,choices=[128,256])
parser.add_argument('-l','--loss',default='bin',choices=['bin','dice'])
parser.add_argument('-heq','--histogram_equalization',dest='heq',action="store_true",default=False)
parser.add_argument('-ct','--contour_type',choices=['i','o','io'],default='i')
parser.add_argument('--yolo',action='store_true', default=False)
parser.add_argument('-ym','--yolomean',action='store_true',default=False)
parser.add_argument('--epoch',type=int,default=40)
parser.add_argument('--test',action='store_true', default=False)
parser.add_argument('--crop_img',action='store_true', default=False)
parser.add_argument('--batch_size','-bs',type=int,default=32)
args = parser.parse_args()

SUNNYBROOK_ROOT_PATH = os.path.abspath(args.filename)
print(SUNNYBROOK_ROOT_PATH)
print('arguments: ',args)
crop_size = args.crop_size
crop_img = args.crop_img
batch_size = args.batch_size
# pre-process image with filter histogram equalization
if args.heq:
    print('enter filter heq')
HEQ = args.heq
contour_type = args.contour_type
yolo = args.yolo
if yolo:
    assert contour_type == 'io',"flag yolo only work with flag ct=='io'"
    print('yolo:', yolo)

shuffle = False
type_unet = args.net
print('net:',type_unet)
print('crop_size :',crop_size)
assert (crop_size ==128 or crop_size ==256),'crop_size has to be 128 or 256'
loss = args.loss
test = args.test
yolo_mean = args.yolomean
### train - save model
name_file = create_name_file(args)
tb_name,_ = name_file.split('.h5')
print('tb_name: ',tb_name)
path_sav = os.path.join(SUNNYBROOK_ROOT_PATH,'save_dir')
full_name = os.path.join(path_sav,name_file)
print('full_name: ',full_name)

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'TrainingDataContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart2',
                   'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart1',
                   'OnlineDataContours')
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_online','challenge_online')
# create a list of object with att case, img_no, ctr_path
#e.g.
# ctr_path = 'E:\\Talles\\NN\\datasets\\sunnybrook\\Sunnybrook Cardiac MR Database ContoursPart3\\TrainingDataContours\\SC-HF-I-02\\contours-manual\\IRCCI-expert\\IM-0001-0167-icontour-manual.txt'
# case = SC-HF-I-2
# img_no = 167 ## look in the path and the number = 0167
print('Mapping ground truth '+contour_type+' contours to images in train...')
if contour_type == 'i' or contour_type == 'o':
    # find image and ground truth label - contour type int or ext
    # crop and center image
    # output array img,mask
    train_ctrs = map_all_contours(TRAIN_CONTOUR_PATH, contour_type, shuffle=False)


    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                crop_size=crop_size,sax=sax)



else:
    # find image and ground truth label - contour type int AND ext
    # crop and center image
    # output array img,mask
    train_ctrs_i = map_all_contours(TRAIN_CONTOUR_PATH, 'i', shuffle=False)
    train_ctrs_o = map_all_contours(TRAIN_CONTOUR_PATH, 'o', shuffle=False)
    io_dict = {}
    for ii in range(len(train_ctrs_o)):
        for each in range(len(train_ctrs_i)):
            if train_ctrs_o[ii].case == train_ctrs_i[each].case and train_ctrs_o[ii].img_no == train_ctrs_i[
                each].img_no:
                io_dict.update({each: ii})
                break
    img_train, mask_train = export_merge_all_contours(train_ctrs_o, train_ctrs_i, TRAIN_IMG_PATH, crop_size, io_dict,sax)

if crop_img:
    # crop images like BB ground truth + 10 pixels
    img_train = make_crop_img (img_train,mask_train,crop_size)

if yolo:
        bb_mask = []
        if mask_train.ndim > 2:
            # modified img here !!!
            mask_train = np.squeeze(mask_train, 3)
        if yolo_mean:
            # condition for images normalization for training in Yolo!!!
            std_img_train, _, _ = mean_std_img(img_train)
            fig_dir = os.path.join('yolo',str(crop_size), 'mean')
        else:
            std_img_train = img_train
            fig_dir = os.path.join('yolo',str(crop_size))
        for ii in range(len(mask_train)):
            bb_mask.append(gera_bb_maskbin(mask_train[ii]))
            filenames_yolo = str('IM.%s.%s.%04d' % (train_ctrs_i[ii].case.replace('-', ''),
                                                    sax[train_ctrs_i[ii].case], train_ctrs_i[ii].img_no))

            xml = YOLO_xml('train',
                           os.path.join(os.getcwd(), fig_dir, 'train', filenames_yolo) + '.png',
                           filenames_yolo,
                           img_train.shape[1:], bb_mask[-1])
            cv2.imwrite(os.path.join(os.getcwd(), fig_dir, 'train', filenames_yolo) + '.png',
                        std_img_train[ii], [cv2.IMWRITE_PNG_COMPRESSION, 4])
            f_xml = open(os.path.join(os.getcwd(), fig_dir, 'annotations', filenames_yolo + '.xml'),
                         'w')
            f_xml.write(xml)
            f_xml.close()
            # data augmentation:
            prob, img, mask = randomShiftScaleRotate(np.squeeze(img_train[ii], 2), mask_train[ii])
            if prob:
                bb_mask.append(gera_bb_maskbin(mask))
                filenames_yolo = str('IM.%s.%s.%04d.aug' % (train_ctrs_i[ii].case.replace('-', ''),
                                                            sax[train_ctrs_i[ii].case], train_ctrs_i[ii].img_no))
                xml = YOLO_xml('train',
                               os.path.join(os.getcwd(), fig_dir, 'train', filenames_yolo) + '.png',
                               filenames_yolo,
                               img_train.shape[1:],
                               bb_mask[-1])
                cv2.imwrite(os.path.join(os.getcwd(), fig_dir,'train', filenames_yolo) + '.png',
                            std_img_train[ii], [cv2.IMWRITE_PNG_COMPRESSION, 4])
                f_xml = open(
                    os.path.join(os.getcwd(), fig_dir, 'annotations', filenames_yolo + '.xml'),
                    'w')
                f_xml.write(xml)
                f_xml.close()
        bb_mask = np.array(bb_mask)
        mask_train = np.expand_dims(mask_train, 3)

if contour_type == 'io':
    # OHE - only for 3 classes
    mask_train = OHE_lbl(mask_train)
split = int(0.1 * len(img_train))
img_dev = img_train[0:split]
mask_dev = mask_train[0:split]
img_train = img_train[split:]
mask_train = mask_train[split:]

# finalized mapping training set
print('Done mapping training set')

### insert filter HEQ

if HEQ:
    img_train = hist_equalize(img_train)
    img_train = np.expand_dims(img_train, -1)
    img_dev = hist_equalize(img_dev)
    img_dev = np.expand_dims(img_dev, -1)
    print('filter HEQ applied to image')

if type_unet != 'FCN' and crop_img==False:
    # normalize images
    x_mean = np.mean(img_train,axis=0)
    x_std = np.std(img_train,axis=0)+1e-5
    img_train -= x_mean
    img_train /= x_std
    img_dev -= x_mean
    img_dev /= x_std

if crop_img:
    x_mean = np.mean(img_train, axis=0)
    img_train -= x_mean
    img_dev -= x_mean

if test:
    assert os.path.isfile(full_name)==True,"don't have this combination of flags yet, please train this combination before try to predict"

    print('Mapping ground truth ' + contour_type + ' contours to images in test...')
    if contour_type == 'i' or contour_type == 'o':
        # find image and ground truth label
        # crop and center image
        # output array img,mask
        if contour_type == 'o':
            test_ctrs_i = map_all_contours(VAL_CONTOUR_PATH, 'i', shuffle=False)
            test_ctrs_o = map_all_contours(VAL_CONTOUR_PATH, 'o', shuffle=False)
            print('\nBuilding Test dataset ...')
            io_dict = {}
            for ii in range(len(test_ctrs_o)):
                for each in range(len(test_ctrs_i)):
                    if test_ctrs_o[ii].case == test_ctrs_i[each].case and test_ctrs_o[ii].img_no == test_ctrs_i[
                        each].img_no:
                        io_dict.update({each: ii})
                        break
                test_ctrs = test_ctrs_o
        else:
            test_ctrs = map_all_contours(VAL_CONTOUR_PATH, contour_type, shuffle=False)
        print('\nBuilding Test dataset ...')
        img_test, mask_test = export_all_contours(test_ctrs,
                                                  VAL_IMG_PATH,
                                                    crop_size=crop_size, sax=sax)
    else:
        print('Mapping ground truth ' + contour_type + ' contours to images in test...')
        test_ctrs_i = map_all_contours(VAL_CONTOUR_PATH, 'i', shuffle=False)
        test_ctrs_o = map_all_contours(VAL_CONTOUR_PATH, 'o', shuffle=False)
        print('\nBuilding Test dataset ...')
        io_dict = {}
        for ii in range(len(test_ctrs_o)):
            for each in range(len(test_ctrs_i)):
                if test_ctrs_o[ii].case == test_ctrs_i[each].case and test_ctrs_o[ii].img_no == test_ctrs_i[
                    each].img_no:
                    io_dict.update({each: ii})
                    break
        img_test, mask_test = export_merge_all_contours(test_ctrs_o, test_ctrs_i, VAL_IMG_PATH, crop_size,
                                                        io_dict, sax)

        # modified here to BB predict YOLO
    if crop_img:
        bb = read_json(str(crop_size))
        # crop images like BB ground truth + 10 pixels
        img_test, mask_test = make_crop_bb_yolo(img_test, mask_test, crop_size,bb)

    if HEQ:
        img_test = hist_equalize(img_test)
        img_test = np.expand_dims(img_test, -1)
        print('filter HEQ applied to image')

    if type_unet != 'FCN':
        # normalize images
        img_test -= x_mean
        if not crop_img:
            img_test /= x_std


if os.path.isfile(full_name):
    print('***Rede Iniciando de onde parou***')
    weights = full_name


input_shape = (crop_size,crop_size)
myunet = UNET_SB(input_shape, contour_type,weights,loss)
if type_unet == 'UNET':
    model = myunet.get_unet()
elif type_unet == 'UNET_NORM':
    model = myunet.get_unet_norm()
elif type_unet == 'FCN':
    model = myunet.fcn_model()
model.summary()
#############
# case contour = io modify this for mask_dev = OHE and net sunny last layer = 3
if not test:
    from keras.preprocessing.image import ImageDataGenerator

    kwargs = dict(
        rotation_range=180,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = args.epoch
    mini_batch_size = batch_size # parametro a ser modificado

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./TB/'+tb_name, histogram_freq=0,
                                             write_graph=True, write_images=True, batch_size=mini_batch_size)

    model.fit_generator(train_generator, steps_per_epoch=len(img_train) / mini_batch_size, epochs=epochs,
                        verbose=1, validation_data=(img_dev,mask_dev),
                        validation_steps=len(img_dev),callbacks=[tbCallBack])

    if not os.path.isdir(path_sav):
        os.makedirs(path_sav)
    model.save(full_name)

# predict images

if test:
    preds_img = model.predict(img_test, batch_size=batch_size, verbose=1)
    if preds_img.shape[3]==1:
        one_layer = True
        preds_img = np.where(preds_img > 0.5, 1, 0) # result final img
        if crop_img:
            preds_img_filtered = preds_img
    else:
        one_layer = False
        preds_img = squeeze_predict(preds_img) # result final img
        if crop_img:
            preds_img_filtered = preds_img
        else:
            mask_pred = preds_img.copy()
            retval,preds_img = cv2.threshold(preds_img,.5,1,cv2.THRESH_BINARY) # transform preds - img binary
            preds_img = preds_img.astype('uint8') # modified dtype preds
    if not crop_img:
        bb = read_json(str(crop_size))
        if contour_type == 'o':
            bb_o = []
            for x in list(io_dict.keys()):
                bb_o.append(bb[x])
            bb = np.array(bb_o)
        # teste
        assert len(bb)==len(preds_img),"n. imgs predict has to b equal n. bb"

        preds_img_filtered = []
        if one_layer:
            preds_img = np.squeeze(preds_img, 3).astype('uint8') # modified dtype preds
        for ii in range(len(bb)):
            preds_img_ = compute_i(preds_img[ii],bb[ii])
            # function for transform coords contour into image
            mask_img = np.zeros([crop_size,crop_size])
            if preds_img_ is not None:
                cv2.fillPoly(mask_img, [preds_img_], 1)
            preds_img_filtered.append(mask_img)
        preds_img_filtered = np.array(preds_img_filtered)
        if not one_layer:
            preds_img_filtered = np.multiply(preds_img_filtered,mask_pred) # multiply img io (binary) mask pred - filter imgs

# calculate dice distance
    mask_test = np.squeeze(mask_test,3)
    mask_test_ohe = OHE_lbl(mask_test)
    preds_img_filtered_ohe = OHE_lbl(preds_img_filtered)
    dists = calc_metric(mask_test_ohe.flatten(),preds_img_filtered_ohe.flatten())
    if mask_test_ohe.shape[-1]>2:
        dist_dice_bg,dist_dice_int,dist_dice_out = dice_loss(mask_test_ohe,preds_img_filtered_ohe)
        print('dice background:', dist_dice_bg)
        print('dice interno:', dist_dice_int)
        print('dice externo:', dist_dice_out)
    else:
        dist_dice_bg,dist_dice_int = dice_loss(mask_test_ohe,preds_img_filtered_ohe)
        print('dice background:', dist_dice_bg)
        print('dice {}: {}'.format(contour_type,dist_dice_int))
    print('f1_zero :',dists[0])
    print('jaccard :',dists[1])
