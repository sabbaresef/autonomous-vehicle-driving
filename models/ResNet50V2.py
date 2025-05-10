from models.model import CIL_Architecture
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Conv2D
from tensorflow.keras.applications import ResNet50V2   
import tensorflow as tf

"""
############################################################################################################################
CILRS with ResNet50V2 as Perception module.

To launch this module you must have installed Tensorflow with at least version 2.0.
#############################################################################################################################
"""

class ResNet50V2_RGB(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(ResNet50V2_RGB,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        perception = ResNet50V2(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(flatten_feature)

        return dense


class ResNet50V2_SingleChannel(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(ResNet50V2_SingleChannel,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        concatenated_input = Concatenate()([input,input,input])
        perception = ResNet50V2(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(concatenated_input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(flatten_feature)

        return dense


class ResNet50V2_Multichannel(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(ResNet50V2_Multichannel,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        # When using the Multichannel network it is necessary to condense RGBD inputs on 3 
        # input channels to the perception backbone.
        convolutional_input = Conv2D(3, (1,1), padding='same', input_shape=self.input_shape, data_format="channels_last")(input)
        perception = ResNet50V2(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(convolutional_input)
        flatten_feature = Flatten()(perception)
        
        # He Normal inizializer for ReLU-activated layers.
        dense = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(flatten_feature)

        return dense