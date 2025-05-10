from version084.carla.agent.cil_agent.models.model import CIL_Architecture
from classification_models.tfkeras import Classifiers
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Conv2D, Conv2DTranspose 
import tensorflow as tf

class ResNet34_RGB(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(ResNet34_RGB,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        ResNet34, _ = Classifiers.get('resnet34')
        perception = ResNet34(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(flatten_feature)

        return dense

class ResNet34_SingleChannel(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(ResNet34_SingleChannel,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        concatenated_input = Concatenate()([input,input,input])
        ResNet34, _ = Classifiers.get('resnet34')
        perception = ResNet34(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(concatenated_input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(flatten_feature)

        return dense

class ResNet34_Multichannel(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(ResNet34_Multichannel,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        convolutional_input = Conv2D(3, (1,1), padding='same', input_shape=self.input_shape, data_format="channels_last")(input)
        ResNet34, _ = Classifiers.get('resnet34')
        perception = ResNet34(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(convolutional_input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(flatten_feature)

        return dense