from version084.carla.agent.cil_agent.models.model import CIL_Architecture
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Conv2D

from tensorflow.keras.applications import InceptionV3     


class InceptionV3_RGB(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(InceptionV3_RGB,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        perception = InceptionV3(input_shape=self.input_shape, include_top=False, weights=self.preload_weights)(input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu')(flatten_feature)

        return dense

class InceptionV3_SingleChannel(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(InceptionV3_SingleChannel,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        concatenated_input = Concatenate()([input,input,input])
        perception = InceptionV3(input_shape=(88,200,3), include_top=False, weights=self.preload_weights)(concatenated_input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu')(flatten_feature)

        return dense

class InceptionV3_Multichannel(CIL_Architecture):

    def __init__(self,input_shape,configuration):
        self.preload_weights = configuration["perception_preload_weights"]
        super(InceptionV3_Multichannel,self).__init__(input_shape,configuration)

    def _model_perception(self,input):
        convolutional_input = Conv2D(3, (1,1), padding='same', input_shape=self.input_shape, data_format="channels_last")(input)
        perception = InceptionV3( include_top=False, weights=self.preload_weights)(convolutional_input)
        flatten_feature = Flatten()(perception)
        dense = Dense(512,activation='relu')(flatten_feature)

        return dense