import os
import sys

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, concatenate, Dropout
from keras.models import Model
import keras.backend as K
from keras import optimizers

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

import config


def BuildUnet(
    num_ch=3,
    shape_patch=config.shape_patch,
    lr=0.001,
    rate_dropout=0.2,
    is_finetune_encoder=False,
    is_finetune_decoder=False
):
    ''' Build U-net model
    Args:
        num_ch : input channel num
        shape_patch : shape of input image
        lr : learning rate
        rate_dropout : rate of dropout
        is_finetune_encoder : fix params of decoder part of network
        is_finetune_decoder : fix params of encoder part of network
    Returns:
        network model
    '''
    def EncodeBlock(x, ch):
        def BaseEncode(x):
            x = BatchNormalization()(x)
            x = Dropout(rate=rate_dropout)(x)
            x = Conv2D(ch, (3, 3), padding='same')(x)
            x = Activation('tanh')(x)
            return x
        
        x = BaseEncode(x)
        x = BaseEncode(x)
        return x

    def DecodeBlock(x, shortcut, ch):
        def BaseDecode(x):
            x = BatchNormalization()(x)
            x = Dropout(rate=rate_dropout)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            x = Activation('tanh')(x)
            return x
        
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, shortcut])

        x = BaseDecode(x)
        x = BaseDecode(x)
        return x
    
    def MSE_Mask(y_true, y_pred):
        gt = y_true[:, :, :, 0]
        mask = y_true[:, :, :, 1]

        # mask = K.cast(mask, 'float32')

        # len_mask = K.sum(mask)
        # is_valid = len_mask > shape_patch[0]*shape_patch[1] * config.threshold_valid
        # mse = K.switch(
        #     is_valid, 
        #     K.sum(K.square(gt - y_pred[:, :, :, 0]) * mask) / len_mask,
        #     0.)

        mse = K.mean(K.square(gt - y_pred[:, :, :, 0]))
        return mse

    input_patch = Input(shape=(*shape_patch, num_ch))
    e0 = Conv2D(8, (1, 1), padding='same')(input_patch)
    e0 = Activation('tanh')(e0)

    e0 = EncodeBlock(e0, 16)

    e1 = AveragePooling2D((2, 2))(e0)
    e1 = EncodeBlock(e1, 32)

    e2 = AveragePooling2D((2, 2))(e1)
    e2 = EncodeBlock(e2, 64)

    e3 = AveragePooling2D((2, 2))(e2)
    e3 = EncodeBlock(e3, 128)

    d2 = DecodeBlock(e3, e2, 64)
    d1 = DecodeBlock(d2, e1, 32)
    d0 = DecodeBlock(d1, e0, 16)

    output_patch = Conv2D(1, (1, 1), padding='same')(d0)

    model = Model(input_patch, output_patch)

    # Fine-tune part of network
    if is_finetune_encoder: # Encoder
        for l in model.layers[38:]:
            l.trainable = False
    elif is_finetune_decoder: # Decoder
        for l in model.layers[:38]:
            l.trainable = False

    adam = optimizers.Adam(lr=lr)
    model.compile(
        optimizer=adam,
        loss=MSE_Mask
    )
    return model