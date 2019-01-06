import os
from utilities.utils import get_SAX_SERIES,center_crop, YOLO_xml,create_name_file
from utilities.manipulate_files import shrink_case, Contour, read_contour, read_merge_contour, map_all_contours,export_all_contours, export_merge_all_contours
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('size',type=int,choices=[128,256])
args = parser.parse_args()
crop_size = args.size
SUNNYBROOK_ROOT_PATH = os.path.abspath('E:/Talles/NN/datasets/sunnybrook')
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart2',
                   'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
sax = get_SAX_SERIES(os.getcwd())
test_ctrs = map_all_contours(VAL_CONTOUR_PATH, 'i', shuffle=False)
# find image and ground truth label
# crop and center image
# output array img,mask

print('\nBuilding Test dataset ...')
img_test, mask_test = export_all_contours(test_ctrs,
                                          VAL_IMG_PATH,
                                          crop_size=crop_size, sax=sax)
for ii in range(len(img_test)):
    filenames = str('IM.%s.%s.%04d' % (test_ctrs[ii].case.replace('-', ''),
                                       sax[test_ctrs[ii].case], test_ctrs[ii].img_no))
    cv2.imwrite(os.path.join(os.getcwd(), 'yolo', str(crop_size), 'test', filenames) + '.png',
                img_test[ii],[cv2.IMWRITE_PNG_COMPRESSION, 4])
