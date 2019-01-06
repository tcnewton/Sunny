from keras.layers import UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ZeroPadding2D, Cropping2D, Conv2DTranspose
from keras import backend as K


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn


def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h // 2, crop_h // 2 + rem_h)
    crop_w_dims = (crop_w // 2, crop_w // 2 + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])

    return cropped


def dice_coef(y_true, y_pred, smooth=0.0, epsilon = 1e-6):
    '''Average dice coefficient per batch.'''
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth + epsilon), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=0.0, epsilon = 1e-6):
    '''Average jaccard coefficient per batch.'''
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean((intersection + smooth) / (union + smooth+epsilon), axis=0)

class UNET_SB(object):

    def __init__(self, crop_size , contour_type,weights,loss):
        self.weights = weights
        self.img_rows = crop_size[0]
        self.img_cols = crop_size[1]
        self.losses = loss
        if contour_type == 'i' or contour_type == 'o':
            self.no_classes = 1
            self.activation = 'sigmoid'
            if self.losses == 'dice':
                self.losses = dice_coef_loss
            else:
                self.losses = 'binary_crossentropy'
        else:
            self.no_classes = 3
            self.losses = 'categorical_crossentropy'
            self.activation = 'softmax'

    def fcn_model(self):
        ''' "Skip" FCN architecture similar to Long et al., 2015
        https://arxiv.org/abs/1411.4038
        '''
        if self.no_classes == 1:
            num_classes = 1
            loss = dice_coef_loss
            activation = 'sigmoid'
        else:
            num_classes = 3
            loss = 'categorical_crossentropy'
            activation = 'softmax'

        kwargs = dict(
            kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
        )

        data = Input(shape=(self.img_rows,self.img_cols,1), dtype='float', name='data')
        mvn0 = Lambda(mvn, name='mvn0')(data)
        pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

        conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
        mvn1 = Lambda(mvn, name='mvn1')(conv1)

        conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
        mvn2 = Lambda(mvn, name='mvn2')(conv2)

        conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
        mvn3 = Lambda(mvn, name='mvn3')(conv3)
        pool1 = MaxPooling2D(pool_size=3, strides=2,
                             padding='valid', name='pool1')(mvn3)

        conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
        mvn4 = Lambda(mvn, name='mvn4')(conv4)

        conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
        mvn5 = Lambda(mvn, name='mvn5')(conv5)

        conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
        mvn6 = Lambda(mvn, name='mvn6')(conv6)

        conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
        mvn7 = Lambda(mvn, name='mvn7')(conv7)
        pool2 = MaxPooling2D(pool_size=3, strides=2,
                             padding='valid', name='pool2')(mvn7)

        conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
        mvn8 = Lambda(mvn, name='mvn8')(conv8)

        conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
        mvn9 = Lambda(mvn, name='mvn9')(conv9)

        conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
        mvn10 = Lambda(mvn, name='mvn10')(conv10)

        conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
        mvn11 = Lambda(mvn, name='mvn11')(conv11)
        pool3 = MaxPooling2D(pool_size=3, strides=2,
                             padding='valid', name='pool3')(mvn11)
        drop1 = Dropout(rate=0.5, name='drop1')(pool3)

        conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
        mvn12 = Lambda(mvn, name='mvn12')(conv12)

        conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
        mvn13 = Lambda(mvn, name='mvn13')(conv13)

        conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
        mvn14 = Lambda(mvn, name='mvn14')(conv14)

        conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
        mvn15 = Lambda(mvn, name='mvn15')(conv15)
        drop2 = Dropout(rate=0.5, name='drop2')(mvn15)

        score_conv15 = Conv2D(filters=num_classes, kernel_size=1,
                              strides=1, activation=None, padding='valid',
                              kernel_initializer='glorot_uniform', use_bias=True,
                              name='score_conv15')(drop2)
        upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                                    strides=2, activation=None, padding='valid',
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    name='upsample1')(score_conv15)
        score_conv11 = Conv2D(filters=num_classes, kernel_size=1,
                              strides=1, activation=None, padding='valid',
                              kernel_initializer='glorot_uniform', use_bias=True,
                              name='score_conv11')(mvn11)
        crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
        fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

        upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                                    strides=2, activation=None, padding='valid',
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    name='upsample2')(fuse_scores1)
        score_conv7 = Conv2D(filters=num_classes, kernel_size=1,
                             strides=1, activation=None, padding='valid',
                             kernel_initializer='glorot_uniform', use_bias=True,
                             name='score_conv7')(mvn7)
        crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
        fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

        upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                                    strides=2, activation=None, padding='valid',
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    name='upsample3')(fuse_scores2)
        crop3 = Lambda(crop, name='crop3')([data, upsample3])
        predictions = Conv2D(filters=num_classes, kernel_size=1,
                             strides=1, activation=activation, padding='valid',
                             kernel_initializer='glorot_uniform', use_bias=True,
                             name='predictions')(crop3)

        model = Model(inputs=data, outputs=predictions)
        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=loss,
                      metrics=['accuracy', dice_coef, jaccard_coef])

        return model

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv9)
        conv10 = Conv2D(self.no_classes ,3, activation=self.activation, padding='same')(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        if self.weights is not None:
            model.load_weights(self.weights)

        model.compile(optimizer=Adam(lr=1e-4), loss=self.losses, metrics=['accuracy'])

        return model

    def get_unet_norm(self):
        inputs = Input((self.img_rows, self.img_cols, 1))
        mvn0 = Lambda(mvn, name='mvn0')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(mvn0)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv1)
        conv1 = Lambda(mvn, name='mvn1')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv2)
        conv2 = Lambda(mvn, name='mvn2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv3)
        conv3 = Lambda(mvn, name='mvn3')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv4)
        conv4 = Lambda(mvn, name='mvn4')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv5)
        conv5 = Lambda(mvn, name='mvn5')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv9)
        conv10 = Conv2D(self.no_classes ,3, activation=self.activation, padding='same')(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        if self.weights is not None:
            model.load_weights(self.weights)
        model.compile(optimizer=Adam(lr=1e-4), loss=self.losses, metrics=['accuracy'])

        return model
