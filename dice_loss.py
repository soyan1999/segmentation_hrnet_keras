from keras import backend as K
import numpy as np
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    dice_weight=np.array([1.0/3,1.0/3,1.0/3],dtype="float32")
    dice_weight=tf.convert_to_tensor(dice_weight,dtype="float32")
    intersection = K.sum(K.abs(y_true * y_pred), axis=(0,1,2))
    result= (2. * intersection + smooth) / (K.sum(K.square(y_true),(0,1,2)) + K.sum(K.square(y_pred),(0,1,2)) + smooth)
    return K.sum(dice_weight*result)
    

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)