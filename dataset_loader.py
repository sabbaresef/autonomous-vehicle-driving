from enum import Flag
import os
import glob
import json
import re
import math
import cv2
import numpy as np
import sys
import logging
import data_parser
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import random
import traceback

from tensorflow.keras.utils import Sequence
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]


def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }
    array = image[:,:,2]
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


class Carla100Generator(Sequence):
    def __init__(self, path="", output="", batch_size=128, hours=15, sensors=["rgb"], speed_factor=12, augmentation=None, balancing=False, shuffle=False, weather=False, weight_sample=None, input_shape=(88,200,3)):
        self.batch_size = batch_size
        self.show_loaded_image = False
        self.input_shape = input_shape

        self.path = path
        self.output = output

        self.hours = hours
        self.sensors = sensors
        self.speed_factor = speed_factor
        self.bins = list(range(-100,101,5))

        self.augmentation = augmentation

        if self.augmentation is None:
            self.augmentation = []

        self.weight_sample = weight_sample
        
        # load dataset policy
        if balancing:
            self.load_balanced_dataset()
        else:
            if shuffle:
                self.load_shuffled_dataset()
            else:
                if weather: 
                    self.load_weather_dataset()
                else:
                    self.load_dataset()
                    

        if self.weight_sample:
            self.analysis(plotting=True)
            

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        # print(np.floor(len(self.access_index)/self.batch_size))
        return int(np.floor(len(self.access_index)/self.batch_size))

    @property
    def shape(self):
        return self.input_shape

    def next(self):
        return self.__next__()

    def __getitem__(self, index):
        start = (index*self.batch_size) % len(self.access_index)
        stop = (start + self.batch_size) % len(self.access_index)

        return self.__get_data(start,stop)
    
    
    def __load_image(self, index, verbose=False):
        preload_image = False

        if verbose:
            start_image_time = datetime.datetime.now()

        if "rgb" in self.sensors:
            if preload_image and 'rgb_data' in self.dataset[index]['input']:
                img_rgb = self.dataset[index]['input']['rgb_data']
            else:
                img_rgb_pass = True
                img_rgb = cv2.imread(self.dataset[index]['input']['rgb'],cv2.IMREAD_COLOR)
                if img_rgb is not None and img_rgb.shape[0] == 88:
                    img_rgb = img_rgb.astype(np.float32)
                    img_rgb = img_rgb / 255.

                    if preload_image:
                        self.dataset[index]['input']['rgb_data'] = img_rgb
                else:
                    img_rgb_pass = False
        
        if verbose:
            end_image_time = datetime.datetime.now()
            delta = end_image_time - start_image_time
            print("Loading RGB Image Time : ", delta)

            start_image_time = datetime.datetime.now()
    
        if "depth" in self.sensors:
            depth_pass = False
            if preload_image and 'depth_data' in self.dataset[index]['input']:
                normalized_depth = self.dataset[index]['input']['depth_data']
                depth_pass = True
            else:
                depth = cv2.imread(self.dataset[index]['input']['depth'],cv2.IMREAD_COLOR)
                if depth is not None:
                    depth = depth.astype(np.float32)
                    normalized_depth = np.dot(depth[:, :, :3], [65536.0, 256.0, 1.0])
                    normalized_depth /= 16777215.0
                    normalized_depth = np.expand_dims(normalized_depth, -1)

                    if preload_image:
                        self.dataset[index]['input']['depth_data'] = normalized_depth
                    depth_pass = True
        
        if verbose:
            end_image_time = datetime.datetime.now()
            delta = end_image_time - start_image_time
            print("Loading Depth Image Time : ", delta)
            
            start_image_time = datetime.datetime.now()
        
        if "segmentation" in self.sensors:
            segmentation_pass = False
            if preload_image and 'segmentation_data' in self.dataset[index]['input']:
                normalized_segmentation = self.dataset[index]['input']['segmentation_data']
                segmentation_pass = True
            else:
                segmentation = cv2.imread(self.dataset[index]['input']['segmentation'],cv2.IMREAD_GRAYSCALE)
                if segmentation is not None:
                    normalized_segmentation = segmentation.astype(np.float32)
                    normalized_segmentation /= 255.0
                    normalized_segmentation = np.expand_dims(normalized_segmentation, -1)
                    segmentation_pass = True
                    if preload_image:
                        self.dataset[index]['input']['segmentation_data'] = normalized_segmentation
        if verbose:
            end_image_time = datetime.datetime.now()
            delta = end_image_time - start_image_time
            print("Loading Segmentation Image Time : ", delta)

        # Combination of shapes
        if verbose:
            start_image_time = datetime.datetime.now()
        multichannel_image = [] 
        if len(self.sensors) > 1:
            num_channels = 3 + (len(self.sensors) - 1)

            if "rgb" in self.sensors and img_rgb_pass:
                multichannel_image.append(img_rgb)
            if "depth" in self.sensors and depth_pass:
                # multichannel_image[:,:,3] = normalized_depth[:,:,0]
                multichannel_image.append(normalized_depth) # np.dstack((multichannel_image, normalized_depth))

            if "segmentation" in self.sensors and segmentation_pass:
                # multichannel_image[:,:,-1] = normalized_segmentation[:,:,0]  # np.dstack((multichannel_image, normalized_segmentation))
                multichannel_image.append(normalized_segmentation)

        elif len(self.sensors) == 1:
            if "rgb" in self.sensors and img_rgb_pass:
                multichannel_image.append(img_rgb)
            elif "depth" in self.sensors and depth_pass:
                multichannel_image.append(normalized_depth)
            elif "segmentation" in self.sensors and segmentation_pass:
                multichannel_image.append(normalized_segmentation)
        
        if verbose:
            end_image_time = datetime.datetime.now()
            delta = end_image_time - start_image_time
            print("Loading Composition Time : ", delta)


        failure = ("rgb" in self.sensors and not img_rgb_pass) or ("depth" in self.sensors and not depth_pass) or ("segmentation" in self.sensors and not segmentation_pass)

        if failure:
            multichannel_image = None
        
        if verbose:
            print("Failure Time : ", failure)

        return multichannel_image

    def __get_data(self,start,stop, verbose=False):
        """
        :param start: Starting index
        :param stop: Stopping index
        :return: (X,y) batch data from dataset loaded
        """
        if verbose:
            start_time = datetime.datetime.now()

        X_rgb = np.empty((self.batch_size,*(88, 200, 3)))
        X_first = np.empty((self.batch_size,*(600, 800, 1)))
        X_second = np.empty((self.batch_size,*(600, 800, 1)))
        
        X_speed = np.empty((self.batch_size))
        X_direction = np.empty((self.batch_size))

        y = np.empty((self.batch_size,5))

        if self.weight_sample:
            sample_weights = np.empty((self.batch_size))

        batch_index = 0

        i = start
        while batch_index < self.batch_size:
            index = self.access_index[i]

            if verbose:
                start_image_time = datetime.datetime.now()
            
            img = self.__load_image(index, verbose=verbose)

            if verbose:
                end_image_time = datetime.datetime.now()
                delta = end_image_time - start_image_time
                print("Loading Image Time : ", delta)

            if img is not None:

                if self.augmentation is not None:
                    pass

                speed = self.dataset[index]['input']['speed'] / self.speed_factor

                # X[batch_index,] = img
                if self.input_shape[2] == 1:
                    X_first[batch_index,] = img[0]
                elif self.input_shape[2] == 3:
                    X_rgb[batch_index,] = img[0]
                elif self.input_shape[2] == 4:
                    X_rgb[batch_index,] = img[0]
                    X_first[batch_index,] = img[1]
                else:
                    X_rgb[batch_index,] = img[0]
                    X_first[batch_index,] = img[1]
                    X_second[batch_index,] = img[2]

                X_speed[batch_index,] = speed
                X_direction[batch_index,] = self.dataset[index]['input']['directions']
                
                y[batch_index,] = [self.dataset[index]['output']['steer'],
                                self.dataset[index]['output']['throttle'],
                                self.dataset[index]['output']['brake'],
                                speed,
                                self.dataset[index]['input']['directions']
                                ]

                if self.weight_sample:
                    weight_bin = min(self.bins, key=lambda y : abs(y - self.dataset[index]['output']['steer']*100))
                    sample_weights[batch_index,] = self.sample_weights[weight_bin]

                batch_index += 1

            i = (i + 1) % len(self.access_index)
        
        # DEBUG
        self.weight_sample = False
        if verbose:
            end_time = datetime.datetime.now()

            delta = end_time - start_time
            print("Loading Data Total Time : ", delta) 

        if self.input_shape[2] == 1:
            return ((X_first, X_speed, X_direction), y, sample_weights) if self.weight_sample else ([X_first, X_speed, X_direction], y)
        elif self.input_shape[2] == 3:
            return ((X_rgb, X_speed, X_direction), y, sample_weights) if self.weight_sample else ([X_rgb, X_speed, X_direction], y)
        elif self.input_shape[2] == 4:
            return ((X_rgb,X_first, X_speed, X_direction), y, sample_weights) if self.weight_sample else ([X_rgb,X_first, X_speed, X_direction], y)
        else:
            return ((X_rgb,X_first,X_second, X_speed, X_direction), y, sample_weights) if self.weight_sample else ([X_rgb,X_first,X_second, X_speed, X_direction], y)
            
        return ((X, X_speed, X_direction), y, sample_weights) if self.weight_sample else ([X, X_speed, X_direction], y)


    def __next__(self):
        start = self.__batch_index*self.batch_size
        stop = start + self.batch_size
        self.__batch_index += 1
        if start > len(self.access_index) - 1 or stop > len(self.access_index) - 1:
            raise StopIteration

        yield self.__get_data(start,stop)


    def analysis(self, plotting=False):
    
        hist = {}

        x = []
        for bin in self.bins:
            hist[bin] = 0

        for sample_number in self.dataset.keys():
            x.append(self.dataset[sample_number]['output']['steer']*100)
            bin = min(self.bins, key=lambda y : abs(y - self.dataset[sample_number]['output']['steer']*100))
            hist[bin] += 1
        
        self.sample_weights = {}

        for bin in hist:
            self.sample_weights[bin] = 1.0 - (float(hist[bin]) / len(self.dataset))
        
        plt.title(f"Steering angle dataset {self.hours} hours", fontsize=16)
        plt.xlabel("Steering angle normalized", fontsize=14)  
        plt.ylabel("Number of steering angles", fontsize=14)
        plt.hist(x, bins=self.bins)
        plt.savefig(os.path.join("graphs", "steer_bins_" + self.output.split("/")[-2] +".pdf"), bbox_inches='tight')
        
        plt.rcdefaults()
        fig, ax = plt.subplots()
        
        plt.title(f"Sample weights dataset {self.hours} hours", fontsize=16)
        plt.xlabel("Steering angle normalized", fontsize=14)  
        plt.ylabel("Number of steering angles", fontsize=14)
        plt.plot(self.bins, list(self.sample_weights.values()))
        plt.savefig(os.path.join("graphs", "steer_sample_" + self.output.split("/")[-2] +".pdf"), bbox_inches='tight')
        
        
        directions = ('Follow Lane', 'Left', 'Right', 'Go Straigh')
        samples_per_directions = [0, 0, 0, 0]

        plt.rcdefaults()
        fig, ax = plt.subplots()
        
        y_pos = np.arange(len(directions))
                
        for sample_number in self.dataset.keys():
            direction = int(self.dataset[sample_number]['input']['directions'])
            samples_per_directions[direction - 2] += 1

        ax.barh(y_pos, samples_per_directions, align='center', color=['blue', 'green', 'red', 'yellow'], height=[0.6, 0.6, 0.6, 0.6], linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(directions)
        ax.invert_yaxis()
        ax.set_title(f"Directions Distribution dataset {self.hours} hours", fontsize=16)
        ax.set_xlabel("Number of samples per direction", fontsize=14)
        ax.set_ylabel("Directions", fontsize=14)
        

        plt.savefig(os.path.join("graphs", "directions_" + self.output.split("/")[-2] +".pdf"), bbox_inches='tight')  
           


    def load_dataset(self):
        """
        Load the dataset from the PLY file.
        If PLY file is not available, else this function creates it.
        """
        
        dataset_name = os.path.join(self.output, 'dataset_' + str(self.hours)+'.npy')
        
        if os.path.exists(dataset_name):
            self.dataset = np.load(dataset_name,allow_pickle=True).item()
            logging.log(logging.INFO,"Loaded Dataset PLY")
        else:
            logging.log(logging.INFO,"Creation Dataset PLY, not balanced, not shuffled")
            episodes_list = glob.glob(os.path.join(self.path, 'episode_*'))
            episodes_list.sort(key=alphanum_key)
            dataset = {}

            i = 0
            current_hours = 0
            sample_number = 0
            logging.info("Loading Dataset")

            pbar = tqdm(total=self.hours)
            while current_hours < self.hours and i < len(episodes_list):
                episode = episodes_list[i]

                measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
                measurements_list.sort(key=alphanum_key)
                
                for measurement in measurements_list:
                    data_point_number = measurement.split('_')[-1].split('.')[0]

                    dataset[sample_number] = {}
                    dataset[sample_number]['input'] = {}
                    dataset[sample_number]['output'] = {}

                    try:

                        with open(measurement) as f:
                            measurement_data = json.load(f)

                        dataset[sample_number]['input']["rgb"] = os.path.join(self.path,episode,'CentralRGB_' + data_point_number + '.png')
                        dataset[sample_number]['input']["depth"] = os.path.join(self.path,episode,'CentralDepth_' + data_point_number + '.png')
                        dataset[sample_number]['input']["segmentation"] = os.path.join(self.path,episode,'CentralSemanticSeg_' + data_point_number + '.png')
                        dataset[sample_number]['input']["lidar"] = os.path.join(self.path,episode,'Lidar_' + data_point_number + '.ply')


                        dataset[sample_number]['input']['speed'] = data_parser.get_speed(measurement_data)
                        dataset[sample_number]['input']['directions'] = measurement_data['directions']

                        dataset[sample_number]['output']['steer'] = measurement_data['steer']
                        dataset[sample_number]['output']['throttle'] = measurement_data['throttle']
                        dataset[sample_number]['output']['brake'] = measurement_data['brake']

                        frame_added = 1
                        
                        if "lateral_cameras" in self.augmentation and np.random.uniform(0, 1) <= 1:
                        
                          if np.random.uniform(0, 1) <= 1:
                              dataset[sample_number + 1] = {}
                              dataset[sample_number + 1]['input'] = {}
                              dataset[sample_number + 1]['output'] = {}
                              dataset[sample_number + 1]['input'] ["rgb"] = os.path.join(self.path,episode,'LeftRGB_' + data_point_number + '.png')
                              dataset[sample_number + 1]['input']["depth"] = os.path.join(self.path,episode,'LeftDepth_' + data_point_number + '.png')
                              dataset[sample_number + 1]['input']["segmentation"] = os.path.join(self.path,episode,'LeftSemanticSeg_' + data_point_number + '.png')
                              dataset[sample_number + 1]['input']['speed'] = data_parser.get_speed(measurement_data)
                              dataset[sample_number + 1]['input']['directions'] = measurement_data['directions']
  
                              dataset[sample_number + 1]['output']['steer'] = self.augment_steering(-30,
                                                                                                  measurement_data['steer'],
                                                                                                  3.6*dataset[sample_number + 1]['input']['speed'],
                                                                                                  6
                                                                                                  )
                              dataset[sample_number + 1]['output']['throttle'] = measurement_data['throttle']
                              dataset[sample_number + 1]['output']['brake'] = measurement_data['brake']
                              frame_added += 1
  
                          if np.random.uniform(0, 1) <= 1:
                              dataset[sample_number + 2] = {}
                              dataset[sample_number + 2]['input'] = {}
                              dataset[sample_number + 2]['output'] = {}
                              dataset[sample_number + 2]['input'] ["rgb"] = os.path.join(self.path,episode,'RightRGB_' + data_point_number + '.png')
                              dataset[sample_number + 2]['input']["depth"] = os.path.join(self.path,episode,'RightDepth_' + data_point_number + '.png')
                              dataset[sample_number + 2]['input']["segmentation"] = os.path.join(self.path,episode,'RightSemanticSeg_' + data_point_number + '.png')
                              dataset[sample_number + 2]['input']['speed'] = data_parser.get_speed(measurement_data)
                              dataset[sample_number + 2]['input']['directions'] = measurement_data['directions']
  
                              dataset[sample_number + 2]['output']['steer'] = self.augment_steering(30,
                                                                                                  measurement_data['steer'],
                                                                                                  3.6*dataset[sample_number + 2]['input']['speed'],
                                                                                                  6
                                                                                                  )
                              dataset[sample_number + 2]['output']['throttle'] = measurement_data['throttle']
                              dataset[sample_number + 2]['output']['brake'] = measurement_data['brake']
                              frame_added += 1
                              
                        sample_number += frame_added

                        FPS_DATASET = 10

                        current_hours += float(frame_added / FPS_DATASET) / 3600
                        pbar.update(float(frame_added / FPS_DATASET) / 3600)
                        print(f"Sample_number:  {sample_number}\n")

                    except BaseException as e:
                        try:
                          import traceback
                          print("Error with "+str(episode)+" "+str(data_point_number))
                          print(traceback.format_exc())
                          print(e, repr(e))
                          pass
                          
                        except:
                          pass
                        
                i += 1
            pbar.close()

            np.save(dataset_name, dataset)
            self.dataset = dataset

        self.access_index = list(self.dataset.keys())



    def load_shuffled_dataset(self):
        """
        Load the dataset from the PLY file. Dataset is shuffled.
        If PLY file is not available, else this function creates it.
        Here a list with all the samples is created, shuffled and then 
        the file is created.
        """
        dataset_name = os.path.join(self.output, 'shuffled_dataset_' + str(self.hours) + '.npy')
        
        if os.path.exists(dataset_name):
            self.dataset = np.load(dataset_name, allow_pickle=True).item()
            logging.log(logging.INFO,"Loaded Dataset Shuffled PLY")
        else:
            dataset_list_name = os.path.join(self.output, 'list_dataset_' + str(self.hours) + '.npy')
            if os.path.exists(dataset_list_name):
                dataset_list = np.load(dataset_list_name, allow_pickle=True).tolist()
                logging.log(logging.INFO,"Loaded Dataset List Shuffled PLY")
            else:
                episodes_list = glob.glob(os.path.join(self.path, 'episode_*'))
                episodes_list.sort(key=alphanum_key)
    
                logging.info("Load Dataset Shuffled")
                dataset_list = []
    
                pbar = tqdm(total=len(episodes_list))
                i = 0
                
                while i < len(episodes_list):          
                    episode = episodes_list[i]
    
                    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
                    measurements_list.sort(key=alphanum_key)
    
                    for measurement in measurements_list:
                        data_point_number = measurement.split('_')[-1].split('.')[0]

                        try:
                            # Load measurements
                            with open(measurement) as f:
                                measurement_data = json.load(f)
                            
                            
                            direction = measurement_data['directions']
                            
                            centralRGB = os.path.join(self.path,episode,'CentralRGB_' + data_point_number + '.png')
                            centralDepth = os.path.join(self.path,episode,'CentralDepth_' + data_point_number + '.png')
                            centralSemanticSeg = os.path.join(self.path,episode,'CentralSemanticSeg_' + data_point_number + '.png')
                            
                            # You first check if all the images are present.
                            if os.path.exists(centralRGB) and os.path.exists(centralDepth) and os.path.exists(centralSemanticSeg):
                                sample = {}
                                sample['input'] = {}
                                sample['output'] = {}
                                
                                sample['input']["rgb"] = centralRGB
                                sample['input']["depth"] = centralDepth
                                sample['input']["segmentation"] = centralSemanticSeg
                                sample['input']['speed'] = data_parser.get_speed(measurement_data)
                                sample['input']['directions'] = direction

                                sample['output']['steer'] = measurement_data['steer']
                                sample['output']['throttle'] = measurement_data['throttle']
                                sample['output']['brake'] = measurement_data['brake']
                
                                dataset_list.append(sample)
                                                 
    
                            if "lateral_cameras" in self.augmentation:
                                leftRGB = os.path.join(self.path,episode,'LeftRGB_' + data_point_number + '.png')
                                leftDepth = os.path.join(self.path,episode,'LeftDepth_' + data_point_number + '.png')
                                leftSemanticSeg = os.path.join(self.path,episode,'LeftSemanticSeg_' + data_point_number + '.png')
                                
                                # You first check if all the images are present.
                                if os.path.exists(leftRGB) and os.path.exists(leftDepth) and os.path.exists(leftSemanticSeg):
                                    sample = {}
                                    sample['input'] = {}
                                    sample['output'] = {}
                                    
                                    sample['input']["rgb"] = leftRGB
                                    sample['input']["depth"] = leftDepth
                                    sample['input']["segmentation"] = leftSemanticSeg
                                    sample['input']['speed'] = data_parser.get_speed(measurement_data)
                                    sample['input']['directions'] = direction

                                    sample['output']['steer'] = self.augment_steering(-30,
                                                                                    measurement_data['steer'],
                                                                                    3.6*sample['input']['speed'],
                                                                                    6)
                                    sample['output']['throttle'] = measurement_data['throttle']
                                    sample['output']['brake'] = measurement_data['brake']
    
                                    dataset_list.append(sample)
    
                            if "lateral_cameras" in self.augmentation:
                                rightRGB = os.path.join(self.path,episode,'RightRGB_' + data_point_number + '.png')
                                rightDepth = os.path.join(self.path,episode,'RightDepth_' + data_point_number + '.png')
                                rightSemanticSeg = os.path.join(self.path,episode,'RightSemanticSeg_' + data_point_number + '.png')
                                
                                # You first check if all the images are present.
                                if os.path.exists(rightRGB) and os.path.exists(rightDepth) and os.path.exists(rightSemanticSeg):
                                    logging.info("Right")
                                    sample = {}
                                    sample['input'] = {}
                                    sample['output'] = {}
                                    
                                    sample['input']["rgb"] = rightRGB
                                    sample['input']["depth"] = rightDepth
                                    sample['input']["segmentation"] = rightSemanticSeg
                                    sample['input']['speed'] = data_parser.get_speed(measurement_data)
                                    sample['input']['directions'] = direction
                                    
                                    sample['output']['steer'] = self.augment_steering(30,
                                                                                    measurement_data['steer'],
                                                                                    3.6*sample['input']['speed'],
                                                                                    6)
                                    sample['output']['throttle'] = measurement_data['throttle']
                                    sample['output']['brake'] = measurement_data['brake']
    
                                    dataset_list.append(sample)
                                                           
                        except Exception as e:
                            import traceback
                            print("Error with " + str(episode) + " " + str(data_point_number))
                            print(traceback.format_exc())
                            print(e, repr(e))
                                            
                    i += 1
                    pbar.update(1) 
    
                pbar.close()
                logging.info("List Created")
    
                random.shuffle(dataset_list)               
                logging.info("List Shuffled")
                
                np.save(dataset_list_name, dataset_list)


            logging.info("Dataset Creation")
            FPS_DATASET = 10
            current_hours = 0

            dataset = {}
            pbar = tqdm(total=self.hours)

            dataset = {}
            sample_number = 0
            number_of_samples = len(dataset_list)
            
            while current_hours < self.hours and sample_number <= number_of_samples - 1:
                dataset[sample_number] = dataset_list[sample_number]

                current_hours += float(1 / FPS_DATASET) / 3600
                pbar.update(float(1 / FPS_DATASET) / 3600)
                
                sample_number += 1
            
            pbar.close()
            logging.info("Dataset Created")

            np.save(dataset_name, dataset)
            self.dataset = dataset

        self.access_index = list(self.dataset.keys())
    
    

    def load_balanced_dataset(self):
        """
        Load the dataset from the PLY file. Dataset is balanced in terms 
        of high-level controls and weather conditions. the dataset is organized 
        so that in a batch there are exactly the same number of samples for each 
        combination of direction and weather condition. Therefore, the batch size 
        must be a multiple of 16. If PLY file is not available, else this function 
        creates it.
        
        """
        dataset_name = os.path.join(self.output, 'balanced_dataset_' + str(self.hours) +'.npy')
        
        if os.path.exists(dataset_name):
            self.dataset = np.load(dataset_name, allow_pickle=True).item()
            logging.log(logging.INFO,"Loaded Dataset Balanced PLY")
        else:
            ordered_dataset_name = os.path.join(self.output, 'ordered_dataset_' + str(self.hours) + '.npy')
            
            if os.path.exists(ordered_dataset_name):
                ordered_dataset = np.load(ordered_dataset_name, allow_pickle=True).item()
                logging.log(logging.INFO,"Loaded Dataset Ordered PLY")
            else:
                episodes_list = glob.glob(os.path.join(self.path, 'episode_*'))
                episodes_list.sort(key=alphanum_key)
    
                logging.info("Ordering Dataset")
                
                # Here a dictionary of dictionaries is used to organize the data in 
                # terms of weather conditions and directions.
                ordered_dataset = {}
    
                # Directions in trainig set
                FOLLOW_LANE = 2
                LEFT = 3
                RIGHT = 4
                GO_STRAIGH = 5
    
                # Weathers in training set
                CLEAR_NOON = 1
                AFTER_RAIN_NOON = 3
                HEAVY_RAIN_NOON = 6
                CLEAR_SUNSET = 8
    
                directions = {}
                directions[FOLLOW_LANE] = []
                directions[LEFT] = []
                directions[RIGHT]= []
                directions[GO_STRAIGH] = []
                ordered_dataset[CLEAR_NOON] = directions
                
                directions = {}
                directions[FOLLOW_LANE] = []
                directions[LEFT] = []
                directions[RIGHT]= []
                directions[GO_STRAIGH] = []
                ordered_dataset[AFTER_RAIN_NOON] = directions
                
                directions = {}
                directions[FOLLOW_LANE] = []
                directions[LEFT] = []
                directions[RIGHT]= []
                directions[GO_STRAIGH] = []
                ordered_dataset[HEAVY_RAIN_NOON] = directions
                
                directions = {}
                directions[FOLLOW_LANE] = []
                directions[LEFT] = []
                directions[RIGHT]= []
                directions[GO_STRAIGH] = []
                ordered_dataset[CLEAR_SUNSET] = directions
    
                pbar = tqdm(total=len(episodes_list))
                i = 0
                frame_added = 0
                
                
                while i < len(episodes_list):          
                    episode = episodes_list[i]
    
                    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
                    measurements_list.sort(key=alphanum_key)
    
                    for measurement in measurements_list:
                        data_point_number = measurement.split('_')[-1].split('.')[0]
                        
                        try:
                            # Load measurements
                            with open(measurement) as f:
                                measurement_data = json.load(f)
                                
                            direction = measurement_data['directions']
                            
                            # Load metadata
                            metadata = glob.glob(os.path.join(episode, 'metadata.json'))[0]
                            with open(metadata) as f:
                                metadata_data = json.load(f)
                            
                            weather = metadata_data['weather']
                               
                            centralRGB = os.path.join(self.path,episode,'CentralRGB_' + data_point_number + '.png')
                            centralDepth = os.path.join(self.path,episode,'CentralDepth_' + data_point_number + '.png')
                            centralSemanticSeg = os.path.join(self.path,episode,'CentralSemanticSeg_' + data_point_number + '.png')
                            
                            # You first check if all the images are present.
                            if os.path.exists(centralRGB) and os.path.exists(centralDepth) and os.path.exists(centralSemanticSeg):
                                sample = {}  
                                sample['input'] = {}
                                sample['output'] = {}
                                
                                sample['input']["rgb"] = centralRGB
                                sample['input']["depth"] = centralDepth
                                sample['input']["segmentation"] = centralSemanticSeg
                                sample['input']['speed'] = data_parser.get_speed(measurement_data)
                                sample['input']['directions'] = direction
                                sample['output']['steer'] = measurement_data['steer']
                                sample['output']['throttle'] = measurement_data['throttle']
                                sample['output']['brake'] = measurement_data['brake']
                                
                                ordered_dataset[weather][direction].append(sample)                        
                                frame_added += 1

                            if "lateral_cameras" in self.augmentation:
                                leftRGB = os.path.join(self.path,episode,'LeftRGB_' + data_point_number + '.png')
                                leftDepth = os.path.join(self.path,episode,'LeftDepth_' + data_point_number + '.png')
                                leftSemanticSeg = os.path.join(self.path,episode,'LeftSemanticSeg_' + data_point_number + '.png')
                                
                                # You first check if all the images are present.
                                if os.path.exists(leftRGB) and os.path.exists(leftDepth) and os.path.exists(leftSemanticSeg):
                                    sample = {}
                                    sample['input'] = {}
                                    sample['output'] = {}
                                    
                                    sample['input']["rgb"] = leftRGB
                                    sample['input']["depth"] = leftDepth
                                    sample['input']["segmentation"] = leftSemanticSeg
                                    sample['input']['speed'] = data_parser.get_speed(measurement_data)
                                    sample['input']['directions'] = direction
                                    sample['output']['steer'] = self.augment_steering(-30,
                                                                                    measurement_data['steer'],
                                                                                    3.6*sample['input']['speed'],
                                                                                    6)
                                    sample['output']['throttle'] = measurement_data['throttle']
                                    sample['output']['brake'] = measurement_data['brake']
                                    
                                    ordered_dataset[weather][direction].append(sample)
                                    frame_added += 1
    
                            if "lateral_cameras" in self.augmentation:
                                rightRGB = os.path.join(self.path,episode,'RightRGB_' + data_point_number + '.png')
                                rightDepth = os.path.join(self.path,episode,'RightDepth_' + data_point_number + '.png')
                                rightSemanticSeg = os.path.join(self.path,episode,'RightSemanticSeg_' + data_point_number + '.png')
                                
                                # You first check if all the images are present.
                                if os.path.exists(rightRGB) and os.path.exists(rightDepth) and os.path.exists(rightSemanticSeg):
                                    sample = {}
                                    sample['input'] = {}
                                    sample['output'] = {}
                                    
                                    sample['input']["rgb"] = rightRGB
                                    sample['input']["depth"] = rightDepth
                                    sample['input']["segmentation"] = rightSemanticSeg
                                    sample['input']['speed'] = data_parser.get_speed(measurement_data)
                                    sample['input']['directions'] = direction
                                    sample['output']['steer'] = self.augment_steering(30,
                                                                                    measurement_data['steer'],
                                                                                    3.6*sample['input']['speed'],
                                                                                    6)
                                    sample['output']['throttle'] = measurement_data['throttle']
                                    sample['output']['brake'] = measurement_data['brake']
                                    
                                    ordered_dataset[weather][direction].append(sample)
                                    frame_added += 1
                                                                
                        except Exception as e:
                            print("Error with " + str(episode) + " " + str(data_point_number))
                            print(traceback.format_exc())
                            print(e, repr(e))
         
                    i += 1
                    pbar.update(1) 
    
                pbar.close()
                logging.info("Dataset Ordered")
                
                total_number_of_samples = 0
                for weather in ordered_dataset.keys():
                    for direction in ordered_dataset[weather].keys():
                        total_number_of_samples += len(ordered_dataset[weather][direction])
                        
                logging.info("\n\nTotal number:  {total_number_of_samples}")

                np.save(ordered_dataset_name, ordered_dataset)

            FPS_DATASET = 10
            current_hours = 0

            logging.info("Balancing Dataset")

            dataset = {}
            
            pbar = tqdm(total=self.hours)
            index = 0
            sample_number = 0
            samples_per_condition = int(self.batch_size / 16) # 16 = num_weathers(4) * num_directions(4)
            
            logging.info(f"samples_per_condition: {samples_per_condition}")
            
            # for each combination we insert batch_size / 16 samples in the dataset. 
            # If the samples are not enough, start sampling again from the beginning 
            # for that combination.
            while current_hours < self.hours:
                logging.info(f"current_hours: {current_hours}")
                for weather in ordered_dataset.keys():
                    for direction in ordered_dataset[weather].keys():
                        for i in range(samples_per_condition):
                            num_samples = len(ordered_dataset[weather][direction])
                            dataset[sample_number] = ordered_dataset[weather][direction][(index + i) % num_samples]
                            sample_number += 1
                            
                            current_hours += float(1 / FPS_DATASET) / 3600
                            pbar.update(float(1 / FPS_DATASET) / 3600)
                            
                index += samples_per_condition # now samples_per_condition samples for each condition have been added.
                
            logging.info(f"\n\n\nsample_number: {sample_number}")

            pbar.close()
            logging.info("Dataset Balanced")

            np.save(dataset_name, dataset)
            self.dataset = dataset
        
        self.access_index = list(self.dataset.keys())

    
    
    def load_weather_dataset(self):
        """
        Load the dataset from the PLY file. Dataset is balanced in terms of 
        episodes for all weather conditions. If PLY file is not available, 
        else this function creates it.
        """
        
        CLEAR_NOON = 1
        AFTER_RAIN_NOON = 3
        HEAVY_RAIN_NOON = 6
        CLEAR_SUNSET = 8
        
        #####################################################################################
        # Useful functions
        def from_index_to_weather(index):
            if index == 0:
                return CLEAR_NOON
            elif index == 1: 
                return AFTER_RAIN_NOON
            elif index == 2:
                return HEAVY_RAIN_NOON
            elif index == 3:
                return CLEAR_SUNSET
            else:
                return None    
    
    
        def episode_weather(episodes):
            """Create a dictionary in which episodes are organized by weather conditions."""
            episode_weather_dict = {CLEAR_NOON:[], AFTER_RAIN_NOON:[], HEAVY_RAIN_NOON:[], CLEAR_SUNSET:[]}
            for episode in episodes:
                try:
                    metadata = glob.glob(os.path.join(episode, 'metadata.json'))[0]
                    with open(metadata) as f:
                        # Load metadata to read weather of episode. 
                        metadata_data = json.load(f)
        
                    weather = metadata_data['weather']
                    episode_weather_dict[weather].append(episode)
                except:
                    logging.info(f"\nNo metadata for episode {episode}!")
            return episode_weather_dict
        #####################################################################################
    
        dataset_name = os.path.join(self.output, 'weather_dataset_' + str(self.hours) + '.npy')
        if os.path.exists(dataset_name):
            self.dataset = np.load(dataset_name, allow_pickle=True).item()
            logging.log(logging.INFO,"Loaded Dataset Weather PLY")
        else:
            if not os.path.exists(self.output):
                os.mkdir(self.output)

            episodes_list = glob.glob(os.path.join(self.path, 'episode_*'))
            episodes_list.sort(key=alphanum_key)
    
            episode_weather_dict = episode_weather(episodes_list)
            
            weather_index = 0
            i = 0
            dataset = {}
            current_hours = 0
            sample_number = 0
    
            logging.info("Loading Dataset")
            pbar = tqdm(total=self.hours)
    
            while current_hours < self.hours:
                weather = from_index_to_weather(weather_index)
                try:
                    episode = episode_weather_dict[weather][i]
                except:
                    random_episode = np.random.randint(0, i-1)
                    episode = episode_weather_dict[weather][random_episode]
                    logging.info(f"No more episode for {weather}, resampling {random_episode}!")
    
                measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
                measurements_list.sort(key=alphanum_key)
    
                for measurement in measurements_list:
                    data_point_number = measurement.split('_')[-1].split('.')[0]
    
                    dataset[sample_number] = {}
                    dataset[sample_number]['input'] = {}
                    dataset[sample_number]['output'] = {}
                    
                    try:
                        with open(measurement) as f:
                            measurement_data = json.load(f)
    
                        dataset[sample_number]['input']["rgb"] = os.path.join(self.path,episode,'CentralRGB_' + data_point_number + '.png')
                        dataset[sample_number]['input']["depth"] = os.path.join(self.path,episode,'CentralDepth_' + data_point_number + '.png')
                        dataset[sample_number]['input']["segmentation"] = os.path.join(self.path,episode,'CentralSemanticSeg_' + data_point_number + '.png')
                        dataset[sample_number]['input']["lidar"] = os.path.join(self.path,episode,'Lidar_' + data_point_number + '.ply')
                        dataset[sample_number]['input']['speed'] = data_parser.get_speed(measurement_data)
                        dataset[sample_number]['input']['directions'] = measurement_data['directions']
    
                        dataset[sample_number]['output']['steer'] = measurement_data['steer']
                        dataset[sample_number]['output']['throttle'] = measurement_data['throttle']
                        dataset[sample_number]['output']['brake'] = measurement_data['brake']
    
                        frame_added = 1
    
                        if "lateral_cameras" in self.augmentation:
                            dataset[sample_number + 1] = {}
                            dataset[sample_number + 1]['input'] = {}
                            dataset[sample_number + 1]['output'] = {}
                            dataset[sample_number + 1]['input'] ["rgb"] = os.path.join(self.path,episode,'LeftRGB_' + data_point_number + '.png')
                            dataset[sample_number + 1]['input']["depth"] = os.path.join(self.path,episode,'LeftDepth_' + data_point_number + '.png')
                            dataset[sample_number + 1]['input']["segmentation"] = os.path.join(self.path,episode,'LeftSemanticSeg_' + data_point_number + '.png')
                            dataset[sample_number + 1]['input']['speed'] = data_parser.get_speed(measurement_data)
                            dataset[sample_number + 1]['input']['directions'] = measurement_data['directions']
    
                            dataset[sample_number + 1]['output']['steer'] = self.augment_steering(-30,
                                                                                                measurement_data['steer'],
                                                                                                3.6*dataset[sample_number + 1]['input']['speed'],
                                                                                                6
                                                                                                )
                            dataset[sample_number + 1]['output']['throttle'] = measurement_data['throttle']
                            dataset[sample_number + 1]['output']['brake'] = measurement_data['brake']
    
                        if "lateral_cameras" in self.augmentation:
                            dataset[sample_number + 2] = {}
                            dataset[sample_number + 2]['input'] = {}
                            dataset[sample_number + 2]['output'] = {}
                            dataset[sample_number + 2]['input'] ["rgb"] = os.path.join(self.path,episode,'RightRGB_' + data_point_number + '.png')
                            dataset[sample_number + 2]['input']["depth"] = os.path.join(self.path,episode,'RightDepth_' + data_point_number + '.png')
                            dataset[sample_number + 2]['input']["segmentation"] = os.path.join(self.path,episode,'RightSemanticSeg_' + data_point_number + '.png')
                            dataset[sample_number + 2]['input']['speed'] = data_parser.get_speed(measurement_data)
                            dataset[sample_number + 2]['input']['directions'] = measurement_data['directions']
    
                            dataset[sample_number + 2]['output']['steer'] = self.augment_steering(30,
                                                                                                measurement_data['steer'],
                                                                                                3.6*dataset[sample_number + 2]['input']['speed'],
                                                                                                6
                                                                                                )
                            dataset[sample_number + 2]['output']['throttle'] = measurement_data['throttle']
                            dataset[sample_number + 2]['output']['brake'] = measurement_data['brake']
    
                        frame_added += 2
                        sample_number += frame_added
    
                        FPS_DATASET = 10
    
                        current_hours += float(frame_added / FPS_DATASET) / 3600
                        pbar.update(float(frame_added / FPS_DATASET) / 3600)
    
                    except Exception as e:
                        import traceback
                        print("Error with "+str(episode)+" "+str(data_point_number))
                        print(traceback.format_exc())
                        print(e, repr(e))
                        
                weather_index = (weather_index + 1) % 4
                if weather_index == 0:
                    i += 1
    
            pbar.close()
    
            np.save(dataset_name, dataset)
            self.dataset = dataset
    
        self.access_index = list(self.dataset.keys())


    def augment_steering(self, camera_angle, steer, speed, AUGMENT_LATERAL_STEERINGS):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))

        return steer


    def augment_measurement(self, measurements, angle, speed, steer_name='steer'):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, measurements[steer_name],
                                          speed)
        measurements[steer_name] = new_steer
        return measurements



if __name__ == '__main__':

    # These are the configurations for the dataset to be loaded.
    configuration = {
        "dataset_path" : "/user/fsabbarese/dataset_unzip/CARLA100/",
        "output_path": "/mnt/sdc1/fsabbarese/test/",        
        "hours" : 15,
        "input_shape" : [88,200,4],
        "train_weight_samples" : True,
        "batch_size" : 128,
        "balancing": True,
        "shuffle": False,
        "weather": False
    }

    dataet_creator = Carla100Generator(     path=configuration['dataset_path'],
                                            output=configuration["output_path"],
                                            batch_size=configuration['batch_size'],
                                            hours=configuration['hours'],
                                            input_shape=configuration["input_shape"],
                                            balancing=configuration["balancing"], 
                                            shuffle=configuration["shuffle"],
                                            weather=configuration["weather"])