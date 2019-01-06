import os,cv2

def get_SAX_SERIES(root_txt_SS):
    SAX_SERIES = {}
    with open(os.path.join(root_txt_SS,'SAX_series.txt'), 'r') as f:
        for line in f:
            if not line.startswith('#'):
                key, val = line.split(':')
                SAX_SERIES[key.strip()] = val.strip()

    return SAX_SERIES

def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    '''
    h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')

    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)  # modificado para ser int
        pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)  # modificado para ser int
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)  # preenche imgs with zeros
        h, w, d = ndarray.shape
    # center crop
    h_offset = (h - crop_size) // 2  # quando a altura da imagem maior que crop_size
    w_offset = (w - crop_size) // 2  # quando a largura da imagem maior que crop_size
    cropped = ndarray[h_offset:(h_offset + crop_size),
              w_offset:(w_offset + crop_size), :]

    return cropped

def YOLO_xml(folder,pathname,filenames,img_shape,bb):
    '''inputs:
    folder - pasta onde ira ficar os arquivos imagens - str
    pathname: path/to/folder - str
    filenames: name of file img -str
    img_shape - (height,width) - tuple
    bb - bounding box [xmin,ymin,xmax,ymax] - array'''

    xml = '<annotation>\n<folder>'+folder+'</folder>\n<filename>'
    xml += filenames+'.png</filename>\n'
    xml += '<path>'+str(pathname)+'</path>\n'
    xml += '<source>\n<database>'+str('Unknown')+'</database>\n</source>\n<size>\n'
    xml += '<width>'+str(img_shape[1])+'</width>\n'+'<height>'+str(img_shape[0])+'</height>\n'
    xml += '<depth>1</depth>\n</size><segmented>0</segmented>\n'
    xml += '<object>\n<name>ROI</name>\n'
    xml += '<bndbox>\n<xmin>'+str(bb[0])+'</xmin>\n'
    xml += '<ymin>'+str(bb[1])+'</ymin>\n'
    xml += '<xmax>'+str(bb[2])+'</xmax>\n'
    xml += '<ymax>'+str(bb[3])+'</ymax>\n</bndbox>\n'
    xml += '<truncated>0</truncated>\n<difficult>0</difficult>\n</object>\n'
    xml += '</annotation>'
    return xml

def create_name_file(*args):
    # pattern to name_of_file savings for several args
    type_unet = args[0].net
    contour_type = args[0].contour_type
    crop_size = args[0].crop_size
    loss = args[0].loss
    HEQ = args[0].heq
    crop_img = args[0].crop_img
    if HEQ:
        name_file = type_unet + '_' + contour_type + '_' + str(crop_size) + '_heq'
    else:
        name_file = type_unet + '_' + contour_type + '_' + str(crop_size)
    if loss == 'bin':
        pass
    else:
        name_file = name_file + '_dice'
    if crop_img:
        name_file = name_file + '_cropped'
    name_file = name_file+'.h5'
    return name_file