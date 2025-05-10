import os
import re
import sys
import glob
import json
import math
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

######################################################################################
# Specify the folder where the dataset is located.
DIR = "/user/fsabbarese/dataset_unzip/CARLA100/"
######################################################################################


# Weathers in training set
CLEAR_NOON = 1
AFTER_RAIN_NOON = 3
HEAVY_RAIN_NOON = 6
CLEAR_SUNSET = 8

# Fps of the dataset CARLA100
FPS_DATASET = 10

# Weathers and Command in dataset CARLA100
weathers = ("Clear Noon", "After Rain Noon", "Heavy Rain Noon", "Clear Sunset")
directions = ('Follow Lane', 'Left', 'Right', 'Go Straigh')




def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]"""
    return [tryint(c) for c in re.split('([0-9]+)', s) ]



def get_speed(measurement_data):
    """ Extract the proper speed from the measurement data dict """

    # If the forward speed is not on the dataset it is because speed is zero.
    if 'playerMeasurements' in measurement_data and \
            'forwardSpeed' in measurement_data['playerMeasurements']:
        return measurement_data['playerMeasurements']['forwardSpeed']
    elif 'velocity_x' in measurement_data:  # We have a 0.9.X data here
        return forward_speed(measurement_data)
    else:  # There is no speed key, probably speed is zero.
        return 0


def from_direction_to_index(direction):
    """Convert a direction into an index"""
    
    return int(direction) - 2


def from_weather_to_index(weather):
    """Convert a weather into an index"""
    
    if weather == CLEAR_NOON:
        return 0
    elif weather == AFTER_RAIN_NOON:
        return 1
    elif weather == HEAVY_RAIN_NOON:
        return 2
    elif weather == CLEAR_SUNSET:
        return 3


def augment_steering(camera_angle, steer, speed, AUGMENT_LATERAL_STEERINGS):
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

        # logging.info('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer


def save_weather_statistics(samples_per_weather, path):
    """Create a figure with weathers statistics"""
    
    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(weathers))

    ax.barh(y_pos, samples_per_weather, align='center', color=['blue', 'green', 'red', 'yellow'], height=[0.6, 0.6, 0.6, 0.6], linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(weathers)
    ax.invert_yaxis()
    ax.set_xlabel('Number of samples per weather', fontsize=14)
    ax.set_ylabel('Weather', fontsize=14)
    ax.set_title('Weather Distribution', fontsize=16)

    plt.savefig(os.path.join("graphs", "weather_" + path.split("/")[-2] + "_ CARLA100_graph" + ".pdf"), bbox_inches='tight') 



def save_directions_statistics(samples_per_direction, path):
    """Create a figure with directions statistics"""
    
    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(directions))
            
    ax.barh(y_pos, samples_per_direction, align='center', color=['blue', 'green', 'red', 'yellow'], height=[0.6, 0.6, 0.6, 0.6], linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(directions)
    ax.invert_yaxis()
    ax.set_xlabel('Number of samples per command', fontsize=14)
    ax.set_ylabel('Command', fontsize=14)
    ax.set_title('Command Distribution', fontsize=16)
    plt.savefig(os.path.join("graphs", "command_" + path.split("/")[-2] + "_ CARLA100_graph" + ".pdf"), bbox_inches='tight')


def save_steer_statistics(steers, bins, path):
    """Create a figure with steer statistics"""
    
    plt.title("Steering angle distribution", fontsize=16)
    plt.xlabel("Steering angle normalized", fontsize=14)  
    plt.ylabel("Number of steering angles", fontsize=14)
    plt.hist(steers, bins=bins)
    plt.savefig(os.path.join("graphs", "steer_bins_" + path.split("/")[-2] + "_ CARLA100_graph" + ".pdf"), bbox_inches='tight')


def statistics_print(samples_per_weather, samples_per_direction):
    """Print distribution weathers and direction distributions"""
    
    original_stdout = sys.stdout
    
    with open('samples_distribution.txt', 'w') as f:
        sys.stdout = f
        
        print("Samples per weather")
        print(weathers[0] + ": " + str(samples_per_weather[0]))
        print(weathers[1] + ": " + str(samples_per_weather[1]))
        print(weathers[2] + ": " + str(samples_per_weather[2]))
        print(weathers[3] + ": " + str(samples_per_weather[3]))
        
        print("\nSamples per direction")
        print(directions[0] + ": " + str(samples_per_direction[0]))
        print(directions[1] + ": " + str(samples_per_direction[1]))
        print(directions[2] + ": " + str(samples_per_direction[2]))
        print(directions[3] + ": " + str(samples_per_direction[3]))      
        
    sys.stdout = original_stdout
    
    

def dataset_analysis(path=''):
    """Compute statistics for weathers and directions.
    Print distribution weathers and direction distributions.
    Generate figures of distributions."""
    
    logging.info(f"Dataset at: {path}\n")

    episodes_list = glob.glob(os.path.join(path, 'episode_*'))
    episodes_list.sort(key=alphanum_key)
    
    logging.info(f"Episode List: {episodes_list}\n\n")
    
    pbar = tqdm(total=len(episodes_list))
    i = 0

    samples_per_weather = [0, 0, 0, 0]
    samples_per_direction = [0, 0, 0, 0]
    
    current_hours = 0
    steers = []

    while i < len(episodes_list):          
        episode = episodes_list[i]

        measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
        measurements_list.sort(key=alphanum_key)
        
        logging.info(f"Processing episode: {i}\n")
               
        for measurement in measurements_list:
            data_point_number = measurement.split('_')[-1].split('.')[0]

            try:
                # Load measurements
                with open(measurement) as f:
                    measurement_data = json.load(f)
                    
                direction = measurement_data['directions']
                speed = get_speed(measurement_data)

                # Load metadata
                metadata = glob.glob(os.path.join(episode, 'metadata.json'))[0]
                with open(metadata) as f:
                    metadata_data = json.load(f)
                
                weather = metadata_data['weather']

                centralRGB = os.path.join(path,episode,'CentralRGB_' + data_point_number + '.png')
                centralDepth = os.path.join(path,episode,'CentralDepth_' + data_point_number + '.png')
                centralSemanticSeg = os.path.join(path,episode,'CentralSemanticSeg_' + data_point_number + '.png')
                    
                if os.path.exists(centralRGB) and os.path.exists(centralDepth) and os.path.exists(centralSemanticSeg):
                    steers.append(measurement_data['steer'] * 100)
                    samples_per_weather[from_weather_to_index(weather)] += 1
                    samples_per_direction[from_direction_to_index(direction)] += 1                   
                    current_hours += float(1 / FPS_DATASET) / 3600
                
                leftRGB = os.path.join(path,episode,'LeftRGB_' + data_point_number + '.png')
                leftDepth = os.path.join(path,episode,'LeftDepth_' + data_point_number + '.png')
                leftSemanticSeg = os.path.join(path,episode,'LeftSemanticSeg_' + data_point_number + '.png')

                if os.path.exists(leftRGB) and os.path.exists(leftDepth) and os.path.exists(leftSemanticSeg):
                    steers.append(augment_steering(-30, measurement_data['steer'], 3.6 * speed, 6) * 100)
                    samples_per_weather[from_weather_to_index(weather)] += 1
                    samples_per_direction[from_direction_to_index(direction)] += 1  
                    current_hours += float(1 / FPS_DATASET) / 3600
            
                rightRGB = os.path.join(path,episode,'RightRGB_' + data_point_number + '.png')
                rightDepth = os.path.join(path,episode,'RightDepth_' + data_point_number + '.png')
                rightSemanticSeg = os.path.join(path,episode,'RightSemanticSeg_' + data_point_number + '.png')

                if os.path.exists(rightRGB) and os.path.exists(rightDepth) and os.path.exists(rightSemanticSeg):                   
                    steers.append(augment_steering(30, measurement_data['steer'],3.6 * speed, 6) * 100)
                    samples_per_weather[from_weather_to_index(weather)] += 1
                    samples_per_direction[from_direction_to_index(direction)] += 1
                    current_hours += float(1 / FPS_DATASET) / 3600
                
            except Exception as e:
                import traceback
                logging.info("Error with " + str(episode) + " " + str(data_point_number))
                logging.info(traceback.format_exc())
                logging.info(e, repr(e))
        
        i += 1
        pbar.update(1)
        print("\n")
        
    logging.info(f"\n\n\n\nCurrent Hours in dataset: {current_hours}")
    
    statistics_print(samples_per_weather, samples_per_direction)
      
    logging.info("\nWeathers statistics")
    save_weather_statistics(samples_per_weather, path)

    logging.info("\nDirections statistics")
    save_directions_statistics(samples_per_direction, path)

    logging.info("\nSteer statistics")
    bins = list(range(-100,101,5))
    save_steer_statistics(steers, bins, path)


if __name__ == '__main__':
    dataset_analysis(DIR)