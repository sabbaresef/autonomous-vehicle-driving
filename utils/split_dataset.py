import os
import glob
import math
import random

##########################################################################################
# Specify the input and output path
INPUT_DIR = "/mnt/sdc1/fsabbarese/dataset_15/training/"
OUTPUT_DIR = "/mnt/sdc1/fsabbarese/to_add_training/"
##########################################################################################


def split_dataset(input_dir, output_dir, split=0.8333):
    """Split the set of input tfrecords into two partitions by 
    moving a randomly selected part to an output directory.
    A percentage of the split must be specified."""
    
    tfrecords_list = glob.glob(os.path.join(input_dir, 'tf_record_*'))
    num_tfrecords = len(tfrecords_list)

    split_1 = int(math.ceil(num_tfrecords * split))
    split_2 = num_tfrecords - split_1
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_2_tfrecords = random.sample(tfrecords_list, split_2)

    for tfrecord in split_2_tfrecords:
        tfrecord_name = tfrecord.split("/")[-1]
        print("Moving: " + tfrecord_name)
        os.rename(tfrecord, os.path.join(output_dir, tfrecord_name))


if __name__ == '__main__':
    split_dataset(INPUT_DIR, OUTPUT_DIR)

