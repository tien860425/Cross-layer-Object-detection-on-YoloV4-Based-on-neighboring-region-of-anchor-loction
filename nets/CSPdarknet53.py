from functools import wraps

from keras import backend as K
from keras.layers import (Add, Concatenate, Conv2D, Layer, MaxPooling2D,
                          UpSampling2D, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

#--------------------------------------------------#
#   單次卷積DarknetConv2D
#   如果步長為2則自己設定padding方式。
#   測試中發現沒有l2正則化效果更好，所以去掉了l2正則化
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷積塊 -> 卷積 + 標準化 + 啟動函數
#   DarknetConv2D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

#--------------------------------------------------------------------#
#   CSPdarknet的結構塊
#   首先利用ZeroPadding2D和一個步長為2x2的卷積塊進行高和寬的壓縮
#   然後建立一個大的殘差邊shortconv、這個大殘差邊繞過了很多的殘差結構
#   主幹部分會對num_blocks進行迴圈，迴圈內部是殘差結構。
#   對於整個CSPdarknet的結構塊，就是一個大殘差塊+內部多個小殘差塊
#--------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    #----------------------------------------------------------------#
    #   利用ZeroPadding2D和一個步長為2x2的卷積塊進行高和寬的壓縮
    #----------------------------------------------------------------#
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)

    #--------------------------------------------------------------------#
    #   然後建立一個大的殘差邊shortconv、這個大殘差邊繞過了很多的殘差結構
    #--------------------------------------------------------------------#
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)

    #----------------------------------------------------------------#
    #   主幹部分會對num_blocks進行迴圈，迴圈內部是殘差結構。
    #----------------------------------------------------------------#
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)

    #----------------------------------------------------------------#
    #   將大殘差邊再堆疊回來
    #----------------------------------------------------------------#
    route = Concatenate()([postconv, shortconv])

    # 最後對通道數進行整合
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

#---------------------------------------------------#
#   CSPdarknet53 的主體部分
#   輸入為一張416x416x3的圖片
#   輸出為三個有效特徵層
#---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3

