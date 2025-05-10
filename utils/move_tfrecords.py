import os
import re
import glob
import numpy as np

####################################################################################
# Paths to specify
PATH_IN="/mnt/sdc1/fsabbarese/dataset_15_balanced/"
PATH_OUT="/mnt/sdc1/fsabbarese/dataset_15_balanced/validation/"
####################################################################################

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


def move(path_in="", path_out="", to_move=15):
    """Move a fixed number of tfrecords from an input directory to an output directory."""
    
    tf_list = glob.glob(os.path.join(path_in, 'tf_record_*'))
    tf_list.sort(key=alphanum_key)

    tf_to_move = tf_list[0:to_move]
    tf_to_move_names = []
    
    for tf in tf_to_move:
        tf_to_move_names.append(tf.split("/")[-1])
    
    for i in range(len(tf_to_move_names)):
        print(f"Moving: {tf_to_move_names[i]}")
        os.rename(tf_to_move[i], os.path.join(path_out, tf_to_move_names[i]))
        
        
        
def move_percentage(path_in="", path_out="", to_move=0.5, starting_index=10, reverse=False):
    """Move a percentage of the tfrecords from an input directory to an output directory."""
    
    tf_list = glob.glob(os.path.join(path_in, 'tf_record_*'))
    tf_list.sort(key=alphanum_key)
    
    if reverse:
        tf_list.reverse()
    
    num_tf_in = len(tf_list)
    print(f"num_tf_in: {num_tf_in}")
    num_tf_out = int(num_tf_in * to_move)
    print(f"num_tf_out: {num_tf_out}")
    
    tf_to_move_names = []
    
    for i in range(starting_index, num_tf_out):
        tf_to_move_names.append(tf_list[i].split("/")[-1])
    
    for i in range(len(tf_to_move_names)):
        os.rename(tf_list[i+starting_index], os.path.join(path_out, tf_to_move_names[i]))
    


if __name__ == '__main__':

    if not os.path.exists(PATH_OUT):
        os.mkdir(PATH_OUT)
        
    move(PATH_IN, PATH_OUT, to_move=2)