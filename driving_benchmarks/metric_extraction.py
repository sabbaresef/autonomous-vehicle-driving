import pandas as pd
import numpy as np
import os
import csv
import json

#####################################################################################################################################################################################
# Indicate the directory in which to find the summary.csv file as output from the benchmark.

DIR = "/user/fsabbarese/driving_benchmarks/_benchmarks_results/Resnet34_RGBD_128_batch_10h_training_5h_balanced_validation_prelu_speed_conv1_sfactor_40_epochs_CARLA100_Town02/"
#####################################################################################################################################################################################

def metrics_extraction(summary_folder):
    """
    This function extracts all the metrics from the summary.csv file: it organizes the number 
    of collisions against pedestrians, vehicles or other, and the number of timeouts by task and 
    weather conditions. It also prints the success. Note this is not the percentages but only the 
    actual number.
    """
    
    summary_file = os.path.join(summary_folder, 'summary.csv')
    summary_data = pd.read_csv(summary_file, delimiter = ',')

    tasks = summary_data['exp_id']
    weathers = summary_data['weather']
    
    results = summary_data['result']
    final_times = summary_data['final_time']
    time_outs = summary_data['time_out']
    end_pedestrian_collisions = summary_data['end_pedestrian_collision']
    end_vehicle_collisions = summary_data['end_vehicle_collision']
    end_other_collisions = summary_data['end_other_collision']
    
    results_matrix = {0: {}, 1: {}, 2: {}}
    for task in tasks:
      for weather in weathers:
        results_matrix[task].update({weather: 0})
    
    time_out_matrix = {0: {}, 1: {}, 2: {}}
    for task in tasks:
      for weather in weathers:
        time_out_matrix[task].update({weather: 0})
        
    end_pedestrian_collision_matrix = {0: {}, 1: {}, 2: {}}
    for task in tasks:
      for weather in weathers:
        end_pedestrian_collision_matrix[task].update({weather: 0})
    
    end_vehicle_collision_matrix = {0: {}, 1: {}, 2: {}}
    for task in tasks:
      for weather in weathers:
        end_vehicle_collision_matrix[task].update({weather: 0})
        
    end_other_collisions_matrix = {0: {}, 1: {}, 2: {}}
    for task in tasks:
      for weather in weathers:
        end_other_collisions_matrix[task].update({weather: 0})
    
    
    for i in range(len(tasks)):
      task = tasks[i].astype(int)
      weather = weathers[i].astype(int)
      result = results[i].astype(int)
      final_time = final_times[i].astype(int)
      time_out = time_outs[i].astype(int)
      end_pedestrian_collision = end_pedestrian_collisions[i].astype(int)
      end_vehicle_collision = end_vehicle_collisions[i].astype(int)
      end_other_collision = end_other_collisions[i].astype(int)
      
      if result == 1 and end_pedestrian_collision == 0 and end_vehicle_collision == 0 and end_other_collision == 0:
        results_matrix[task][weather] += 1
        
      if result == 0 and end_pedestrian_collision > 0:
        end_pedestrian_collision_matrix[task][weather] += 1
      
      if result == 0 and end_vehicle_collision > 0 and end_pedestrian_collision == 0:
        end_vehicle_collision_matrix[task][weather] += 1
      
      if result == 0 and end_other_collision > 0 and end_pedestrian_collision == 0 and end_vehicle_collision == 0:
        end_other_collisions_matrix[task][weather] += 1
      
      if result ==  0 and time_out == final_time and end_pedestrian_collision == 0 and end_vehicle_collision == 0 and end_other_collision == 0:
        time_out_matrix[task][weather] += 1
    
    with open(os.path.join(summary_folder, 'metrics_extraction.json'), 'w') as fp:
      json.dump('Results: ', fp)
      json.dump(results_matrix, fp)
      json.dump('Collision Pedestrian: ', fp)
      json.dump(end_pedestrian_collision_matrix, fp)
      json.dump('Collision Vehicle: ', fp)
      json.dump(end_vehicle_collision_matrix, fp)
      json.dump('Collision Other: ', fp)
      json.dump(end_other_collisions_matrix, fp)
      json.dump('Time_out: ', fp)
      json.dump(time_out_matrix, fp)
      

if __name__ == '__main__':
    metrics_extraction(DIR)
