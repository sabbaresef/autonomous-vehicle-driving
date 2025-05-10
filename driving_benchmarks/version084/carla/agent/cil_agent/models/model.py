from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Activation, Dense, Concatenate, Flatten, Lambda, PReLU
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Model
import sys
import numpy as np

import tensorflow as tf
import datetime


# Direction for CILRS model
FOLLOW_LANE = 2
LEFT = 3
RIGHT = 4
GO_STRAIGH = 5



"""
############################################################################################################################
Metrics and Loss functions

#############################################################################################################################
"""

def MSE_metric(y_true, y_pred):
    """The calculation Mean squared error"""
    
    direction = tf.cast(y_pred[:,13],dtype=tf.int32)

    batch_size = y_true.shape[0]

    results = tf.constant(0,shape=(batch_size,3),dtype=tf.float32)

    for i in range(2, 6):
        cond = tf.cast(direction == tf.constant(i,shape=(batch_size,),dtype=tf.int32), dtype=tf.float32)
        batch_cond = tf.stack([cond,cond,cond],1)
        results = tf.add(results, tf.multiply(y_pred[:,3*(i-2):3*(i-1)],batch_cond))
    
    error_vector = tf.square(results[:,:] - y_true[:,:3])
    error_mean = tf.reduce_mean(error_vector)

    return error_mean


def MAE_metric(y_true, y_pred):
    """The calculation Mean absolute error"""
    
    direction = tf.cast(y_pred[:,13],dtype=tf.int32)

    batch_size = y_true.shape[0]

    results = tf.constant(0,shape=(batch_size,3),dtype=tf.float32)

    for i in range(2, 6):
        cond = tf.cast(direction == tf.constant(i,shape=(batch_size,),dtype=tf.int32), dtype=tf.float32)
        batch_cond = tf.stack([cond,cond,cond],1)
        results = tf.add(results, tf.multiply(y_pred[:,3*(i-2):3*(i-1)],batch_cond))
    
    error_vector = tf.abs(results[:,:] - y_true[:,:3])
    error_mean = tf.reduce_mean(error_vector)

    return error_mean


def L1_loss(y_true, y_pred, loss_weights=[0.5,0.5,0.5,0.5,0.95], batch_size=32):
    """The calculation of the L1 loss"""
    
    direction = tf.cast(y_pred[:,13],dtype=tf.int32)
    speed = y_pred[:,12]

    batch_size = y_true.shape[0]

    results = tf.constant(0,shape=(batch_size,3),dtype=tf.float32)

    for i in range(2, 6):
        cond = tf.cast(direction == tf.constant(i,shape=(batch_size,),dtype=tf.int32), dtype=tf.float32)
        batch_cond = tf.stack([cond,cond,cond],1)
        results = tf.add(results, tf.multiply(y_pred[:,3*(i-2):3*(i-1)],batch_cond))

    
    error_vector = tf.abs(results[:,:] - y_true[:,:3])
    speed_error_vector = tf.abs(speed[:] - y_true[:,3])

    error_vector = tf.tensordot(error_vector,loss_weights[:3],1)

    error_mean = tf.reduce_mean(error_vector)
    speed_error_mean = tf.reduce_mean(speed_error_vector)

    error = loss_weights[4]*error_mean + loss_weights[3]*speed_error_mean

    return error


def L1_loss_distribution_weight(y_true, y_pred, loss_weights=[0.5,0.5,0.5,0.5,0.95], batch_size=32):
    """The calculation of the L1 loss with distribution weight.
    
    This is similar to the L1 loss, but a weight derived from 
    the distribution of the directions of the samples in the 
    batches is applied. The greater the number of samples and the 
    lower the weight, in order to reduce the influence of that 
    direction in the calculation of the overall loss.
    """
    
    batch_size = y_true.shape[0]
    direction = tf.cast(y_pred[:,13],dtype=tf.int32)
    y, idx, count = tf.unique_with_counts(direction)
    direction_weights = [1,1,1,1]
    
    # The weight for each direction is calculated as the opposite 
    # of the number of samples for the direction normalized for the batch size.
    for i in range(len(y)):
        if y[i] == FOLLOW_LANE:
            direction_weights[0] = float(1 - count[i] / batch_size)
        if y[i] == LEFT:
            direction_weights[1] = float(1 - count[i] / batch_size)
        if y[i] == RIGHT:
            direction_weights[2] = float(1 - count[i] / batch_size)
        if y[i] == GO_STRAIGH:
            direction_weights[3] = float(1 - count[i] / batch_size)
    
    speed = y_pred[:,12]
    results = tf.constant(0,shape=(batch_size,3),dtype=tf.float32)
    batch_weights = tf.constant(0,shape=batch_size,dtype=tf.float32)

    for i in range(2, 6):
        cond = tf.cast(direction == tf.constant(i,shape=(batch_size,),dtype=tf.int32), dtype=tf.float32)
        batch_cond = tf.stack([cond,cond,cond],1)
        mult = tf.multiply(y_pred[:,3*(i-2):3*(i-1)],batch_cond)
        results = tf.add(results, mult)
        batch_weights = tf.add(batch_weights, tf.multiply(cond, direction_weights[i - 2])) # We assign a weight to each sample based on the direction

    error_vector = tf.abs(results[:,:] - y_true[:,:3])
    speed_error_vector = tf.abs(speed[:] - y_true[:,3])
    error_vector = tf.tensordot(error_vector, loss_weights[:3],1)
    error_vector = tf.multiply(error_vector, batch_weights) # We apply the weights assigned to the batch samples.
    error_mean = tf.reduce_mean(error_vector)
    speed_error_mean = tf.reduce_mean(speed_error_vector)
    error = loss_weights[4]*error_mean + loss_weights[3]*speed_error_mean
    
    return error




"""
############################################################################################################################
CILRS Architecture

For ReLU-activated layers He Normal inizializer. 

Actually we have:
 - Tanh for steer action;
 - PReLU for throttle and brake
 - PReLU for speed prediction. 
#############################################################################################################################
"""
class CIL_Architecture:
    FOLLOW_LANE = 2
    LEFT = 3
    RIGHT = 4
    GO_STRAIGH = 5

    def __init__(self, input_shape, configuration):

        self.input_shape = input_shape
        self.model = self.__model_creation()
        weights = configuration["weights"]
        batch_size = configuration["batch_size"]

        loss_functions = lambda x,y: L1_loss(x,y,weights)

        if configuration["optimizer"] == 'adam':
            optimizer = tf.keras.optimizers.Adam(configuration["learning_rate"])
        
        if configuration["training"]:
          self.model.compile(optimizer=optimizer,loss=loss_functions,metrics=[MAE_metric,MSE_metric])

    def _model_perception(self,input):
        conv = Conv2D(32,5,strides=2)(input)
        bn = BatchNormalization()(conv)
        dropout = Dropout(0.0)(bn)
        activation = Activation('relu')(dropout)
        flatten_feature = Flatten()(activation)

        return flatten_feature

    def __model_creation(self, version = 1):
        num_channels = self.input_shape[2]
        input_layers = []
        concatenation_layers = []

        if version == 1:
            input_layer_image = Input(shape=self.input_shape)
            multichannel_input = input_layer_image

            input_layers = [multichannel_input]

        elif version == 2:
            if num_channels == 1:
                input_layer_image_firstch = Input(shape=(None,None,1))
                resized_firstch = Lambda(lambda image : tf.image.resize(image, [self.input_shape[0], self.input_shape[1]]))(input_layer_image_firstch)
                multichannel_input = resized_firstch
                input_layers.append(input_layer_image_firstch)
            elif num_channels == 3:
                input_layer_image_rgb = Input(shape=(None,None,3))
                resized_rgb = Lambda(lambda image : tf.image.resize(image, [self.input_shape[0], self.input_shape[1]]))(input_layer_image_rgb)
                multichannel_input = resized_rgb
                input_layers.append(input_layer_image_rgb)
            else:
                input_layer_image_rgb = Input(shape=(None,None,3))
                resized_rgb = Lambda(lambda image : tf.image.resize(image, [self.input_shape[0], self.input_shape[1]]))(input_layer_image_rgb)
                input_layer_image_firstch = Input(shape=(None,None,1))
                resized_firstch = Lambda(lambda image : tf.image.resize(image, [self.input_shape[0], self.input_shape[1]]))(input_layer_image_firstch)
                
                input_layers.append(input_layer_image_rgb)
                input_layers.append(input_layer_image_firstch)
                
                concatenation_layers.append(resized_rgb)
                concatenation_layers.append(resized_firstch)
            
                if num_channels == 5:
                    input_layer_image_secondch = Input(shape=(None,None,1))
                    resized_secondch  = Lambda(lambda image : tf.image.resize(image, [self.input_shape[0], self.input_shape[1]]))(input_layer_image_secondch)
                    concatenation_layers.append(resized_secondch)
                    input_layers.append(input_layer_image_secondch)

                multichannel_input = Concatenate(axis=3)(concatenation_layers)

        input_layer_measurement = Input(1)
        input_direction = Input(1)

        perception = self._model_perception(multichannel_input)

        feat_measure = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(input_layer_measurement)
        feat_measure = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(feat_measure)

        concatenate = Concatenate()([perception,feat_measure])
        join = Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(concatenate)
        
        straight_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(join)
        straight_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(straight_branch)
        straight_branch = Dropout(0.5)(straight_branch)
        #straight_branch = Dense(3,name='straight_branch')(straight_branch)
        """
        steer_straight_branch = Dense(1, activation='tanh')(straight_branch)
        throttle_straight_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(straight_branch)
        brake_straight_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(straight_branch)
        straight_branch = Concatenate()([steer_straight_branch, throttle_straight_branch, brake_straight_branch])
        """
        
        steer_straight_branch = Dense(1, activation='tanh')(straight_branch)
        throttle_straight_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(straight_branch)
        throttle_straight_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(throttle_straight_branch)
        brake_straight_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(straight_branch)
        brake_straight_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(brake_straight_branch)
        straight_branch = Concatenate()([steer_straight_branch, throttle_straight_branch, brake_straight_branch])
        
        
        left_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(join)
        left_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(left_branch)
        left_branch = Dropout(0.5)(left_branch)
        #left_branch = Dense(3,name='left_branch')(left_branch)
        """
        steer_left_branch = Dense(1, activation='tanh')(left_branch)
        throttle_left_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(left_branch)
        brake_left_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(left_branch)
        left_branch = Concatenate()([steer_left_branch, throttle_left_branch, brake_left_branch])
        """
        
        steer_left_branch = Dense(1, activation='tanh')(left_branch)
        throttle_left_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(left_branch)
        throttle_left_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(throttle_left_branch)
        brake_left_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(left_branch)
        brake_left_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(brake_left_branch)
        left_branch = Concatenate()([steer_left_branch, throttle_left_branch, brake_left_branch])
        
        
        right_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(join)
        right_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(right_branch)
        right_branch = Dropout(0.5)(right_branch)
        #right_branch = Dense(3,name='right_branch')(right_branch)
        """
        steer_right_branch = Dense(1, activation='tanh')(right_branch)
        throttle_right_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(right_branch)
        brake_right_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(right_branch)
        right_branch = Concatenate()([steer_right_branch, throttle_right_branch, brake_right_branch])
        """
        
        steer_right_branch = Dense(1, activation='tanh')(right_branch)
        throttle_right_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(right_branch)
        throttle_right_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(throttle_right_branch)
        brake_right_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(right_branch)
        brake_right_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(brake_right_branch)
        right_branch = Concatenate()([steer_right_branch, throttle_right_branch, brake_right_branch])
        

        follow_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(join)
        follow_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(follow_branch)
        follow_branch = Dropout(0.5)(follow_branch)
        #follow_branch = Dense(3,name='follow_branch')(follow_branch)
        """
        steer_follow_branch = Dense(1, activation='tanh')(follow_branch)
        throttle_follow_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(follow_branch)
        brake_follow_branch = Dense(1, activation='relu', kernel_initializer=self.initializer)(follow_branch)
        follow_branch = Concatenate()([steer_follow_branch, throttle_follow_branch, brake_follow_branch])
        """
        
        steer_follow_branch = Dense(1, activation='tanh')(follow_branch)
        throttle_follow_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(follow_branch)
        throttle_follow_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(throttle_follow_branch)
        brake_follow_branch = Dense(1, kernel_initializer=tf.keras.initializers.he_normal())(follow_branch)
        brake_follow_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(brake_follow_branch)
        follow_branch = Concatenate()([steer_follow_branch, throttle_follow_branch, brake_follow_branch])
        
        
        speed_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(join)
        speed_branch = Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(speed_branch)
        speed_branch = Dropout(0.5)(speed_branch)
        speed_branch = Dense(1,name='speed_branch', kernel_initializer=tf.keras.initializers.he_normal())(speed_branch)
        speed_branch = PReLU(alpha_initializer=Constant(value=0.5), alpha_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))(speed_branch)
        
        output = Concatenate()([follow_branch,left_branch,right_branch,straight_branch,speed_branch, input_direction])

        model = Model(inputs=(*input_layers,input_layer_measurement,input_direction), outputs=output)
        
        return model


    def predict(self, X):
        direction = tf.cast(X[2],dtype=tf.int32)
        batch_size = direction.shape[0]

        y = self.model.predict(X)
 
        results = tf.constant(0,shape=(batch_size,3),dtype=tf.float32)

        for i in range(2, 6):
            cond = tf.cast(direction == tf.constant(i,shape=(batch_size,),dtype=tf.int32), dtype=tf.float32)
            batch_cond = tf.stack([cond,cond,cond],1)
            results = tf.add(results, tf.multiply(y[:,3*(i-2):3*(i-1)],batch_cond))

        return results.numpy()
