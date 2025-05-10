from version084.carla.agent.agent import Agent
from version084.carla.client import VehicleControl

from version084.carla.agent.cil_agent.models.RGBmobilenet import MobileNetV2_RGB, MobileNetV2_SingleChannel, MobileNetV2_Multichannel
from version084.carla.agent.cil_agent.models.InceptionNetBackend import InceptionV3_RGB, InceptionV3_SingleChannel, InceptionV3_Multichannel
from version084.carla.agent.cil_agent.models.ResNet50V2 import ResNet50V2_RGB, ResNet50V2_SingleChannel, ResNet50V2_Multichannel
from version084.carla.agent.cil_agent.models.ResNet34 import ResNet34_RGB, ResNet34_SingleChannel, ResNet34_Multichannel
from version084.carla.image_converter import labels_to_array, depth_to_array, to_bgra_array

from skimage.transform import resize
from tensorflow.keras import Model

import tensorflow as tf
import numpy as np
import glob
import cv2
import os
import logging
import time


#################################################################################
# Set to True if you want to observe the agent while driving.
SHOW = False
#################################################################################


class CILAgent(Agent):
	
    def __init__(self, configuration):
        self.frame_count = 2
        self.speed_factor = configuration["speed_factor"]
        self.sensors = configuration["sensors"]

        self.num_channels = 0

        if "rgb" in configuration["sensors"]:
            self.num_channels = 3

        if "depth" in configuration["sensors"]:
            self.num_channels += 1

        if "segmentation" in configuration["sensors"]:
            self.num_channels += 1

        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print("Device:", tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except:
            strategy = tf.distribute.get_strategy()
        print("\nNumber of replicas:", strategy.num_replicas_in_sync)

        netName = configuration["net"] + "_" + configuration["model"]

        if netName == "MobileNetV2_RGB" :
            self.cil_architecture = MobileNetV2_RGB(configuration["input_shape"],configuration)
        elif netName == "MobileNetV2_SingleChannel" :
            self.cil_architecture = MobileNetV2_SingleChannel(configuration["input_shape"],configuration)
        elif netName == "MobileNetV2_Multichannel" :
            self.cil_architecture = MobileNetV2_Multichannel(configuration["input_shape"],configuration)

        elif netName == "InceptionV3_RGB":
            self.cil_architecture = InceptionV3_RGB(configuration["input_shape"],configuration)
        elif netName == "InceptionV3_SingleChannel":
            self.cil_architecture = InceptionV3_SingleChannel(configuration["input_shape"],configuration)
        elif netName == "InceptionV3_Multichannel":
            self.cil_architecture = InceptionV3_Multichannel(configuration["input_shape"],configuration)
        
        elif netName == "ResNet50V2_RGB":
            self.cil_architecture = ResNet50V2_RGB(configuration["input_shape"],configuration)
        elif netName == "ResNet50V2_SingleChannel":
            self.cil_architecture = ResNet50V2_SingleChannel(configuration["input_shape"],configuration)
        elif netName == "ResNet50V2_Multichannel":
            self.cil_architecture = ResNet50V2_Multichannel(configuration["input_shape"],configuration)
        
        elif netName == "ResNet34_RGB":
            self.cil_architecture = ResNet34_RGB(configuration["input_shape"],configuration)
        elif netName == "ResNet34_SingleChannel":
            self.cil_architecture = ResNet34_SingleChannel(configuration["input_shape"],configuration)
        elif netName == "ResNet34_Multichannel":
            self.cil_architecture = ResNet34_Multichannel(configuration["input_shape"],configuration)
        
        print(f"\nNetwork name: {netName}")
        
        weights_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights", configuration["net"], configuration["model"], configuration["experiment_name"] + ".hdf5")

        if os.path.exists(weights_file_path):
            self.cil_architecture.model.load_weights(weights_file_path)
            print("\nModel loaded correctly !!!")
        else:
            weights_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights", configuration["net"], configuration["model"], configuration["experiment_name"] + ".h5")
            if os.path.exists(weights_file_path):
                self.cil_architecture.model.load_weights(weights_file_path)
                print("\nModel loaded correctly!!!")
            else:
                print(weights_file_path)
                print("\nError, no weights exists!!!")
                exit(0)
        
        print("\nModel ready!!!\n")


    def _process_model_outputs(self, outputs):
        """
        Check the correctness of the commands and add a bit of heuristics in the control, 
        to eventually make car faster, for instance.
        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        if throttle < 0:
            throttle = 0
        if throttle > 1:
            throttle = 1

        if brake < 0.05:
            brake = 0.0
        elif brake < 0:
            brake = 0
        elif brake > 1:
            brake = 1

        if throttle > brake:
            brake = 0.0

        return steer, throttle, brake


    def _process_sensors(self, sensor_data):
        """
        The images are processed in order to obtain an input consistent with 
        what the agent was trained with.
        """
        
        image_BGR = None
        if 'rgb' in self.sensors:
            if sensor_data.get("CentralRGB", None) is not None:
                image_BGRA = to_bgra_array(sensor_data["CentralRGB"])
                image_BGR = cv2.cvtColor(image_BGRA, cv2.COLOR_BGRA2BGR)
                image_BGR = image_BGR.astype(np.float32)
                image_BGR = image_BGR / 255.
                
                #####################################
                if SHOW: 
                    self.frame_count += 1
                    if self.frame_count % 2== 0:
                        cv2.namedWindow("bgr", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('bgr', 400, 300) 
                        cv2.imshow("bgr", image_BGR)
                        cv2.waitKey(1)
                        self.frame_count = 0
                #####################################
                
                if image_BGR.shape != (88,200,3):
                    image_BGR = image_BGR[90:(90+395), :, :]
                    image_BGR = resize(image_BGR, (88,200,3))
            else:
                print("No camera RGB sensor data available!!!")
                exit(0)
        
        depth = None
        if 'depth' in self.sensors:
            if sensor_data.get("CentralDepth", None) is not None:
                depth = depth_to_array(sensor_data["CentralDepth"])
                if depth.shape != (88,200):
                    depth = depth[90:(90+395), :]
                    depth = resize(depth, (88,200))
            else:
                print("No depth sensor data available!!!")
                exit(0)

        semantic = None
        if 'semantic' in self.sensors:
            if sensor_data.get("CentralSemanticSeg", None) is not None:
                semantic = labels_to_array(sensor_data["CentralSemanticSeg"])
                semantic = semantic.astype(np.float32)
                semantic /= 12.0
                if semantic.shape != (88,200):
                    semantic = semantic[90:(90+395), :]
                    semantic = resize(semantic, (88,200,1))
            else:
                print("No camera semantic segmentation sensor data available!!!")
                exit(0)

        multichannel_image = np.zeros((88,200,self.num_channels), np.float32)
        
        if 'rgb' in self.sensors:
            multichannel_image[:,:,:3] = image_BGR
        
        if 'depth' in self.sensors:
            if self.num_channels > 3:
                multichannel_image[:,:,3] = depth[:,:]
            else:
                multichannel_image[:,:,0] = depth[:,:]
        
        if 'semantic' in self.sensors:
            if self.num_channels == 5:
                multichannel_image[:,:,-1] = semantic[:,:]
            elif self.num_channels == 4:
                multichannel_image[:,:,3] = semantic[:,:]
            elif self.num_channels == 2:
                multichannel_image[:,:,-1] = semantic[:,:]
            else:
                multichannel_image[:,:,0] = semantic[:,:]
        
        multichannel_image = tf.convert_to_tensor(multichannel_image)
        multichannel_image = tf.expand_dims(multichannel_image, axis=0)

        return multichannel_image


    def _process_measurements(self, measurements):
        """
        The speed is normalized based on the speed factor with which the model was trained.
        """
        
        player = measurements.player_measurements
        forward_speed = player.forward_speed
        forward_speed_normalized = forward_speed / self.speed_factor
        forward_speed_normalized = tf.constant(forward_speed_normalized, dtype=tf.float32, shape=(1, ))

        return forward_speed_normalized
    
    
    def _clear_session(self):
        tf.keras.backend.clear_session()
        

    def run_step(self, measurements, sensor_data, directions, target):
        """
        This function must be implemented for an agent running on CARLA. 
        Implement the agent's driving policy. It receives the inputs and 
        generates the steer, throttle and brake actuation commands.
        """
        
        forward_speed_normalized = self._process_measurements(measurements)
        multichannel_image = self._process_sensors(sensor_data)
        directions = tf.constant(directions, dtype=tf.float32, shape=(1, ))
        cil_input = [multichannel_image, forward_speed_normalized, directions]        
        cil_output = self.cil_architecture.predict(cil_input)
                
        steer, throttle, brake = self._process_model_outputs(cil_output[0])

        control = VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        
        self._clear_session()      
        
        return control


