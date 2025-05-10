import cv2
import os
import glob
import time
import numpy as np
import skimage as sk
import tensorflow as tf
import matplotlib.pyplot as plt

# Libraries for data augmentation.
import augmentation
import Automold

from skimage.transform import resize
from functools import partial
from tqdm import tqdm
from dataset_loader import Carla100Generator

# tf.config.experimental_run_functions_eagerly(True)

# Weathers in training set
FOLLOW_LANE = 2.
LEFT = 3.
RIGHT = 4.
GO_STRAIGH = 5.


#########################################################################################################################
# Useful functions. 
def _float_feature_array(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#########################################################################################################################
#
#
#
#########################################################################################################################
# Data augmentation

"""
def data_augmentation(image, augmentation=False, prob_augmentation=0.3):
    # Weak data augmentation
    if augmentation and np.random.uniform(0, 1) < prob_augmentation:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if np.random.uniform(0, 1) < prob_augmentation:

            image = sk.color.rgb2hsv(image)
            image[:, :, 1] = np.clip(image[:, :, 1] *  np.random.uniform(0.75, 1.25) +  np.random.uniform(0.007, 0.13), 0, 1)
            image = sk.color.hsv2rgb(image)
            image = np.clip(image, 0, 1)
        
        image = np.uint8(image*255)

        if np.random.uniform(0, 1) < prob_augmentation:
            no_of_shadows = np.random.randint(1, 2)
            shadow_dimension = np.random.randint(3, 10)
            image = Automold.add_shadow(image, no_of_shadows, shadow_dimension=shadow_dimension)
        
        if np.random.uniform(0, 1) < prob_augmentation:
            if np.random.uniform(0, 1) < 0.5:
                darkness_coeff = np.random.uniform(0, 0.8)
                image = Automold.darken(image, darkness_coeff)
            else:
                brightness_coeff = np.random.uniform(0, 0.8)
                image = Automold.brighten(image, brightness_coeff)
        
        if np.random.uniform(0, 1) < prob_augmentation:
            fog_coeff = np.random.uniform(0.2, 0.35)
            image = Automold.add_fog(image, fog_coeff)
        
        image = np.float32(image)
        image = image/255.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

"""


"""
def data_augmentation(image, augmentation=False, prob_augmentation=0.3):
    # Medium data augmentation
    
    if augmentation and np.random.uniform(0, 1) < prob_augmentation:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if np.random.uniform(0, 1) < prob_augmentation:
            image = sk.color.rgb2hsv(image)
            image[:, :, 1] = np.clip(image[:, :, 1] *  np.random.uniform(0.80, 1.20) +  np.random.uniform(0.007, 0.13), 0, 1)
            image = sk.color.hsv2rgb(image)
            image = np.clip(image, 0, 1)
        
        image = np.uint8(image * 255)

        if np.random.uniform(0, 1) < prob_augmentation:
            no_of_shadows = np.random.randint(1, 4)
            shadow_dimension = np.random.randint(3, 10)
            image = Automold.add_shadow(image, no_of_shadows, shadow_dimension=shadow_dimension)
        
        if np.random.uniform(0, 1) < prob_augmentation:
            random = np.random.uniform(0, 1)
            if random <= 0.33:
                darkness_coeff = np.random.uniform(0, 0.70)
                image = Automold.darken(image, darkness_coeff)
                
            elif random > 0.33 and random <= 0.66:
                brightness_coeff = np.random.uniform(0, 0.70)
                image = Automold.brighten(image, brightness_coeff)   
            else:
                darkness_coeff = np.random.uniform(0, 0.70)
                brightness_coeff = np.random.uniform(0, 0.70)
                if np.random.uniform(0, 1) <= 0.5:
                    split = np.random.randint(30, image.shape[0] - 30)
                    image_split_augmented = Automold.brighten(image[0:split, :, :], brightness_coeff)
                    image[0:split, :, :] = image_split_augmented
                    image_split_augmented = Automold.darken(image[split+1: , :, :], darkness_coeff)
                    image[split+1: , :, :] = image_split_augmented
                else:
                    split = np.random.randint(30, image.shape[1] - 30)
                    image_split_augmented = Automold.brighten(image[:, 0:split, :], brightness_coeff)
                    image[:, 0:split, :] = image_split_augmented
                    image_split_augmented = Automold.darken(image[:, split+1: , :], darkness_coeff)
                    image[:, split+1: , :] = image_split_augmented
        
        if np.random.uniform(0, 1) < prob_augmentation:
            fog_coeff = np.random.uniform(0.20, 0.35)
            image = Automold.add_fog(image, fog_coeff)
        
        image = np.float32(image)
        image = image/255.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

"""


def data_augmentation(image, augmentation=False, prob_augmentation=0.3):
    """Strong data augmentation"""
    
    if augmentation and np.random.uniform(0, 1) < prob_augmentation:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        # Random hue variation
        if np.random.uniform(0, 1) < prob_augmentation:
            image = sk.color.rgb2hsv(image)
            image[:, :, 1] = np.clip(image[:, :, 1] *  np.random.uniform(0.80, 1.20) +  np.random.uniform(0.007, 0.13), 0, 1)
            image = sk.color.hsv2rgb(image)
            image = np.clip(image, 0, 1)
        
        image = np.uint8(image*255)
        
        # Add gravel to simulate puddles
        if np.random.uniform(0, 1) < prob_augmentation - 0.1:
            x1=0
            y1=int(image.shape[0]*2/5)
            x2=image.shape[1]
            y2=image.shape[0]
            rectangular_roi=(x1, y1, x2, y2)
            no_of_patches = np.random.randint(4, 10)
            image = Automold.add_gravel(image, rectangular_roi, no_of_patches)
        
        # Add shadow
        if np.random.uniform(0, 1) < prob_augmentation:
            no_of_shadows = np.random.randint(1, 4)
            shadow_dimension = np.random.randint(3, 7)
            image = Automold.add_shadow(image, no_of_shadows, shadow_dimension=shadow_dimension)
        
        # Add fog or rain (exclusively)
        if np.random.uniform(0, 1) < prob_augmentation - 0.05:
            # Fog
            if np.random.uniform(0, 1) <= 0.5:
                fog_coeff = np.random.uniform(0.2, 0.32)
                image = Automold.add_fog(image, fog_coeff)
                
            # Rain
            else:
                slant = np.random.randint(0, 3)
                drop_length = np.random.randint(1, 3)
                color = np.random.randint(180, 200)
                drop_color = (color, color, color)
                image = Automold.add_rain(image, slant=slant, drop_length=drop_length, drop_color=drop_color)   
        
        # Darken, Brighten or Darken&Brighten (exclusively)
        if np.random.uniform(0, 1) < prob_augmentation:    
            random = np.random.uniform(0, 1)
            
            # Darken
            if random <= 0.33:
                darkness_coeff = np.random.uniform(0, 0.80)
                image = Automold.darken(image, darkness_coeff)
            
            # Brighten
            elif random > 0.33 and random <= 0.66:
                brightness_coeff = np.random.uniform(0, 0.80)
                image = Automold.brighten(image, brightness_coeff)   
                
            # Darken&Brighten applied horizontally or vertically.
            else:
                darkness_coeff = np.random.uniform(0, 0.80)
                brightness_coeff = np.random.uniform(0, 0.80)
                if np.random.uniform(0, 1) <= 0.5:
                    split = np.random.randint(30, image.shape[0] - 30)
                    image_split_augmented = Automold.brighten(image[0:split, :, :], brightness_coeff)
                    image[0:split, :, :] = image_split_augmented
                    image_split_augmented = Automold.darken(image[split+1: , :, :], darkness_coeff)
                    image[split+1: , :, :] = image_split_augmented
                else:
                    split = np.random.randint(30, image.shape[1] - 30)
                    image_split_augmented = Automold.brighten(image[:, 0:split, :], brightness_coeff)
                    image[:, 0:split, :] = image_split_augmented
                    image_split_augmented = Automold.darken(image[:, split+1: , :], darkness_coeff)
                    image[:, split+1: , :] = image_split_augmented   

        image = np.float32(image)
        image = image/255.
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    return image


def flip_augmentation(image, steer, direction, flip=False, prob_flip=0.3, flip_all=False):
    """ 
    This function applies the flip augmentation, with which the perception part 
    is flipped horizontally together with an inversion of the steering and in the 
    case of the Turn Left and Turn Right commands a switch is made between them.
    """
    
    if flip and (direction == LEFT or direction == RIGHT or flip_all):
        
        random = tf.numpy_function(np.random.uniform, [0,1], tf.double)
    
        if random <= prob_flip:
            image = tf.image.flip_left_right(image)
        
            if direction == LEFT:
                direction = RIGHT
        
            if direction == RIGHT:
                direction == LEFT
            
            steer = - steer
            
    return image, steer, direction
#########################################################################################################################
#
#
#
#########################################################################################################################
# Tfrecords parsing

def read_tfrecord(example, num_channels=5, sensors=[], speed_factor = 12.0, augmentation=False, prob_augmentation=0.3, flip=False, prob_flip=0.3):
    """
    This function is used to parse the tfrecords in order to organize the samples in 
    a way suitable for the CILRS model. If necessary, apply the data augmentation.
    """
    tfrecord_format = { 
        'image': tf.io.VarLenFeature(tf.float32),
        'speed':  tf.io.FixedLenFeature([], tf.float32), 
        'direction': tf.io.FixedLenFeature([], tf.float32), 

        'steer': tf.io.FixedLenFeature([], tf.float32),  
        'throttle':tf.io.FixedLenFeature([], tf.float32), 
        'brake': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example, tfrecord_format)
    dense_tensor = tf.sparse.to_dense(parsed['image'])
    image = tf.reshape(dense_tensor, (88,200,5))
    
    if num_channels >= 3:
        # Only if the number of channels is 3 the data augmentation is applied.
        RGB_image = tf.numpy_function(data_augmentation, [image[:, :, :3], augmentation, prob_augmentation], tf.float32)
    if num_channels == 1 and 'depth' in sensors:
        image = image[:,:,3]
    elif num_channels == 1 and 'segmentation' in sensors:
        image = image[:,:,4]
    if num_channels == 3:
        image = RGB_image
    elif num_channels == 4 and 'depth' in sensors:
        image = tf.concat([RGB_image, tf.expand_dims(image[:,:,3], -1)], axis=2)
    elif num_channels == 4 and 'segmentation' in sensors:
        image = tf.concat([RGB_image, tf.expand_dims(image[:,:,4], -1)], axis=2)
    
    speed = tf.cast(parsed["speed"], tf.float32)
    speed = speed / speed_factor
    direction = tf.cast(parsed["direction"], tf.float32)
    steer = tf.cast(parsed["steer"], tf.float32)
    throttle = tf.cast(parsed["throttle"], tf.float32)
    brake = tf.cast(parsed["brake"], tf.float32)
    
    if num_channels >= 3:
        # Only if the number of channels is at least 3 the flip augmentation is applied.
        image, steer, direction = flip_augmentation(image, steer, direction, flip, prob_flip, flip_all=False)
     
    return ((image, speed, direction), [steer, throttle, brake, speed, direction])
#########################################################################################################################
#
#
#
#########################################################################################################################
# Dataset Loader

class DatasetLoader():
    """
    Passing some configuration it loads the dataset.
    """
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE 
    def __init__(self, configuration, training=False):
        self.batch_size = configuration["batch_size"] if training else configuration["val_batch_size"]
        self.speed_factor = configuration["speed_factor"]
        
        # We apply data augmentation only in trainin case
        if training:
            self.augmentation = configuration["augmentation"]
            self.prob_augmentation = configuration["prob_augmentation"]
            self.flip = configuration["flip"]
            self.prob_flip = configuration["prob_flip"]
        else:
            self.augmentation = False
            self.prob_augmentation = 0.
            self.flip = False
            self.prob_flip = 0.

        self.num_channels = 0
        
        # The number of channels is defined on the basis of the sensors.
        if "rgb" in configuration["sensors"]:
            self.num_channels = 3
        if "depth" in configuration["sensors"]:
            self.num_channels += 1
        if "segmentation" in configuration["sensors"]:
            self.num_channels += 1

        self.sensors = configuration["sensors"]
        self.balancing = configuration["balancing"]
        self.training = training


    def load_dataset(self,filenames):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # Disable order, increase speed
        
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # Automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order)  # Uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(partial(read_tfrecord, 
                                        num_channels=self.num_channels, 
                                        sensors=self.sensors, 
                                        speed_factor=self.speed_factor,
                                        augmentation=self.augmentation, 
                                        prob_augmentation=self.prob_augmentation,
                                        flip=self.flip,
                                        prob_flip=self.prob_flip), 
                                        num_parallel_calls=DatasetLoader.AUTOTUNE
        )
        return dataset
    

    
    def get_dataset(self, filenames):
        """
        This function implements the input pipeline to the model.
        """
        # Dataset is loaded
        dataset = self.load_dataset(filenames)
        
        # In training case, we shuffle samples. 
        if self.training:
            if self.balancing:
                # If the dataset is balanced the shuffle is done with a small buffer to avoid ruining the sorting of the dataset.
                dataset = dataset.shuffle(buffer_size=int(self.batch_size / 2))
            else:
                dataset = dataset.shuffle(buffer_size=self.batch_size * 10)
        
        # Let's reduce the dataset into batches.
        dataset = dataset.batch(self.batch_size)
        
        # In training case, we shuffle batches.
        if self.training: 
            if self.balancing:
                dataset = dataset.shuffle(buffer_size=13)
            else:
                dataset = dataset.shuffle(buffer_size=3)
        
        dataset = dataset.prefetch(buffer_size=DatasetLoader.AUTOTUNE)
        return dataset  
#########################################################################################################################
#
#
#
#########################################################################################################################
# Dataset Converter
    
class DatasetConverter():
    """ 
    This class creates datasets in tfrecords format.
    """
    def __init__(self, dataset_path, dataset_name):
        self.dataset_path = dataset_path
        try:
            self.dataset = np.load(dataset_path+dataset_name+".npy",allow_pickle=True).item()
            self.access_index = list(self.dataset.keys())
        except Exception:
            raise Exception("Errors in loading the dataset")
    
    
    def __load_image(self, index):
        """
        This function loads images. These are first normalized between 0 and 1, and then 
        a crop of the sky and of the lower part is carried out, and, therefore, a resize 
        to the size of 200 x 88 to be passed as input to the model. This process is 
        applied for RGB, depth and semantic segmentation images.
        """
        
        ### RGB
        preload_image = False
        if preload_image and 'rgb_data' in self.dataset[index]['input']:
            img_rgb = self.dataset[index]['input']['rgb_data']
        else:
            img_rgb = cv2.imread(self.dataset[index]['input']['rgb'],cv2.IMREAD_COLOR)
            if img_rgb is not None:
                img_rgb = img_rgb.astype(np.float32)
                img_rgb = img_rgb / 255. # normalization
                if img_rgb.shape != (88,200,3):
                    img_rgb = img_rgb[90:(90+395), :, :] # crop
                    img_rgb = resize(img_rgb, (88,200,3)) # resize

                if preload_image:
                    self.dataset[index]['input']['rgb_data'] = img_rgb

        depth_pass = False
        if preload_image and 'depth_data' in self.dataset[index]['input']:
            normalized_depth = self.dataset[index]['input']['depth_data']
            depth_pass = True
        else:
            depth = cv2.imread(self.dataset[index]['input']['depth'],cv2.IMREAD_COLOR)
            if depth is not None:
                depth = depth.astype(np.float32)
                # Depth images need to be converted into distances. These operations are 
                # found in the CARLA simulator documentation.
                normalized_depth = np.dot(depth[:, :, :3], [65536.0, 256.0, 1.0]) # normalization
                normalized_depth /= 16777215.0 # normalization

                if normalized_depth.shape != (88,200):
                    normalized_depth = normalized_depth[90:(90+395), :] # crop
                    normalized_depth = resize(normalized_depth, (88,200,1)) # resize

                if preload_image:
                    self.dataset[index]['input']['depth_data'] = normalized_depth
                depth_pass = True
        
        segmentation_pass = False
        if preload_image and 'segmentation_data' in self.dataset[index]['input']:
            normalized_segmentation = self.dataset[index]['input']['segmentation_data']
            segmentation_pass = True
        else:
            segmentation = cv2.imread(self.dataset[index]['input']['segmentation'], cv2.IMREAD_GRAYSCALE)
            if segmentation is not None:
                # In CARLA simulator we have 12 classes for semantic segmentation. 
                normalized_segmentation = segmentation.astype(np.float32)
                normalized_segmentation /= 12.0 # normalization
                
                if normalized_segmentation.shape != (88,200):
                    normalized_segmentation = normalized_segmentation[90:(90+395), :] # crop
                    normalized_segmentation = resize(normalized_segmentation, (88,200,1)) # resize

                segmentation_pass = True
                if preload_image:
                    self.dataset[index]['input']['segmentation_data'] = normalized_segmentation
        
        # Combination of shapes
        num_channels = 5

        multichannel_image = np.zeros((88,200,num_channels), np.float32)
        multichannel_image[:,:,:3] = img_rgb
        if depth_pass:
            multichannel_image[:,:,3] = normalized_depth[:,:,0]

        if segmentation_pass:
            multichannel_image[:,:,-1] = normalized_segmentation[:,:,0]  
        
        failure = not depth_pass or not segmentation_pass

        if failure:
            tf.print(f"Failure at index: {index}")
            multichannel_image = None

        return multichannel_image


    def convert_dataset(self, tfrecord_filepath, max_examples_per_record = 1280):
        """
        This function creates tfrecords starting from a numpy file.
        """
        
        i = 0
        tf_record_count = 0

        if not os.path.exists(tfrecord_filepath):
            os.mkdir(tfrecord_filepath)

        tfrecord_filename = tfrecord_filepath + 'tf_record_{}.tfrecords'
        
        # We also use a progress bar
        progess_bar = tqdm(range(len(self.access_index)))
        
        # All elements within the dataset are processed.
        while i < len(self.access_index):
            with tf.io.TFRecordWriter(tfrecord_filename.format(tf_record_count)) as writer:
                for example_count in range(max_examples_per_record):

                    if i >= len(self.access_index):
                        break

                    index = self.access_index[i]
                    multichannel_image = self.__load_image(index)

                    if multichannel_image is None:
                        i += 1
                        example_count -= 1
                        continue

                    h,w,c = multichannel_image.shape

                    multichannel_image_reshape = np.reshape(multichannel_image, h*w*c)
                    
                    speed = self.dataset[index]['input']['speed']
                    direction = self.dataset[index]['input']['directions']
                    steer = self.dataset[index]['output']['steer']
                    throttle = self.dataset[index]['output']['throttle']
                    brake = self.dataset[index]['output']['brake']
                  
                    feature = { 
                        'image': _float_feature_array(multichannel_image_reshape),
                        'speed': _float_feature(speed), 
                        'direction': _float_feature(direction), 

                        'steer': _float_feature(steer), 
                        'throttle': _float_feature(throttle),  
                        'brake': _float_feature(brake)
                    }

                    # Create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # Writing the serialized example.
                    writer.write(example.SerializeToString())
                    
                    i += 1
                    progess_bar.update(1)
            
            tf_record_count += 1
                
#########################################################################################################################
#
#
#
#########################################################################################################################
if __name__ == "__main__":
    convert = True

    if convert:
        DIR = "/mnt/sdc1/fsabbarese/dataset_15_balanced/"
        dataset_converter = DatasetConverter(DIR, "balanced_dataset_15")
        dataset_converter.convert_dataset(DIR, max_examples_per_record = 1280)

    else:
        import json
        with open("./configuration_mobilenet/configs_test.json",'r') as f:
            configuration = json.load(f)
            
        dataset_loader = DatasetLoader(configuration)
        dataset_loader = DatasetLoader(configuration, training=True)
        training_files = glob.glob("/mnt/sdc1/fsabbarese/dataset_15/training/*.tfrecords")
        training_generator = dataset_loader.get_dataset(training_files)

        for e in training_generator.take(10000):
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 400, 300) 
            cv2.imshow("image", e[0][0].numpy()[0,:, :, 0:3].astype(np.float32))
            cv2.waitKey(1000)
        
        
        
        
        
        
