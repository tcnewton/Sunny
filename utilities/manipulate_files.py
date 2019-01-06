import re, pydicom, cv2, os, fnmatch
import numpy as np
from utilities.utils import center_crop

def shrink_case(case):
    toks = case.split('-')
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return '-'.join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        ctr_path = re.sub(r'\\', '/', ctr_path)  # line included
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)

    __repr__ = __str__


def read_contour(contour, data_path,SAX_SERIES):
    filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)  # name of file e.g.
    full_path = os.path.join(data_path, contour.case, filename)
    f = pydicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]

    return img, mask

def read_merge_contour(contour_o,contour_i, data_path,SAX_SERIES):
    filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour_i.case], contour_i.img_no)  # name of file e.g.
    full_path = os.path.join(data_path, contour_i.case, filename)
    f = pydicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour_o.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 2)
    coords = np.loadtxt(contour_i.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    return img, mask

def map_all_contours(contour_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(contour_path)
                for f in fnmatch.filter(files,
                                        'IM-0001-*-' + contour_type + 'contour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = list(map(Contour, contours))  # include list - function map modified python v.3 reurns generator

    return contours


def export_all_contours(contours, data_path, crop_size,sax):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path,sax)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks

def export_merge_all_contours(contour_o,contours_i, data_path, crop_size,io_dict,sax):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours_i)))
    images = np.zeros((len(contours_i), crop_size, crop_size, 1))
    masks = np.zeros((len(contours_i), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours_i):
        if idx in io_dict.keys():
            o_idx = io_dict[idx]
            img, mask = read_merge_contour(contour_o[o_idx],contour, data_path,sax)
        else:
            img, mask = read_contour(contour, data_path, sax)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask
    return images, masks
