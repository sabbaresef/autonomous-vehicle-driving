import os
import glob
import re
import cv2 as cv
import time

################################################################################################################
# To specify to find capture directory
MODEL = "Resnet50V2_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor"
TOWN = "Town02"
NAME = "Capture_27"
################################################################################################################
# If you want to print the information in the video, the causes of failure and successes must be entered here.


# 'Collision with pedestrian'
# 'Collision with vehicle'
# 'Collisione with other'
# 'Causal Confusion'
# 'Completed'

results = ['Completed', 'Collision with pedestrian', 'Completed', 'Completed', 'Causal Confusion', 'Collision with vehicle']

################################################################################################################

weather_name_dict = {1: 'Training weather: Clear Noon', 3: 'Training weather: After Rain Noon', 6: 'Training weather: Heavy Rain Noon', 8: 'Training weather: Clear Sunset', 10: 'Test weather: Rainy After Rain', 14: 'Test weather: Soft Rain Sunset'}

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
    


def produce_video(model='', town='', name=''):
    """This feature reads the frames from the _images folder and creates the video in the _videos folder."""
    
    capture_directory = os.path.join('_benchmarks_results', name + '_' + model + '_' + 'Capture_' + town)
    input_directory = os.path.join(capture_directory, '_images')
    output_directory = os.path.join(capture_directory, '_videos')
    
    video_name = os.path.join(output_directory, name + '.mp4')
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 70
    writer = cv.VideoWriter(video_name, fourcc, fps, (800, 600))
    font = cv.FONT_HERSHEY_COMPLEX
    
    while not os.path.exists(output_directory):
        os.mkdir(output_directory)
        time.sleep(2)
    
    episodes = glob.glob(input_directory + '/*')
    episodes.sort(key=alphanum_key)
    
    print(f"Episodes: {episodes}\n\n")
    
    i = 0
    for episode in episodes:
        print(f"Episode dir: {episode}")
        weather = episode.split('/')[-1].split('_')[-3]
        print(f"Weather ID: {weather}")
        weather = weather_name_dict[int(weather)]
        print(f"Weather: {weather}")
        
        RGB_directory = os.path.join(episode, 'CentralRGB')
        frames = glob.glob(RGB_directory + '/*')
        frames.sort(key=alphanum_key)
        
        for frame in frames:
            frame = frame = cv.imread(frame)
            cv.putText(frame, weather, (50, 50), font, 1, (0, 255, 255), 2)
            
            if len(results) > 0:
              cv.putText(frame, 'Result: ' + results[i] , (50, 90), font, 1, (0, 255, 255), 2)
              
            writer.write(frame)
            
        print(f"Finish episode{episode}\n\n")
        i += 1
    
    cv.destroyAllWindows()
    writer.release()
    

if __name__ == '__main__':
    produce_video(model=MODEL, town=TOWN, name=NAME)