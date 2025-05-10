import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import os
import math
import random

############################################################################################################
# Set dataset folder for which make validation of tfrecords and split into training and validation sets

DIR = "/user/fsabbarese/dataset_15_balanced/"
#DIR = "/mnt/sdc1/fsabbarese/dataset_15/"
############################################################################################################


def validate_dataset(dir, reader_opts=None, remove=False):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    :param remove: (optional) if True is used to remove corrupted tfrecords. 
    """
    i = 0
    filenames = glob.glob(dir + "/*.tfrecords")
    
    for fname in filenames:
        print('\nvalidating ', fname)
        
        if remove and os.path.getsize(fname) == 0:
            os.remove(fname)
            continue

        record_iterator = tf.io.tf_record_iterator(path=fname, options=reader_opts)
        try:
            for _ in record_iterator:
                i += 1
        except Exception as e:
            print('\nerror in {} at record {}'.format(fname, i))
            print(e)
            if remove:
                os.remove(fname)


def split_dataset(dir, training=0.80):
    """Split the set of input tfrecords into two partitions by 
    moving a randomly selected part to an output directory.
    A percentage of the split must be specified."""
    
    tfrecords_list = glob.glob(DIR + "/*.tfrecords")
    
    training_dir = os.path.join(dir, "training")
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    validation_dir = os.path.join(dir, "validation")
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    
    num_tfrecords = len(tfrecords_list)

    training_split = int(math.ceil(num_tfrecords * training))
    validation_split = num_tfrecords - training_split

    validation_split_tfrecords = random.sample(tfrecords_list, validation_split)

    for tfrecord in validation_split_tfrecords:
        tfrecord_name = tfrecord.split("/")[-1]
        print("Moving to validation folder: " + tfrecord_name)
        os.rename(tfrecord, os.path.join(validation_dir, tfrecord_name))
    
    tfrecords_list = glob.glob(dir + "/*.tfrecords")
    
    for tfrecord in tfrecords_list:
        tfrecord_name = tfrecord.split("/")[-1]
        print("Moving to training folder: " + tfrecord_name)
        os.rename(tfrecord, os.path.join(training_dir, tfrecord_name))
    

if __name__ == '__main__':

    validate_dataset(DIR, reader_opts=None, remove=True)
    split_dataset(DIR, training=0.80)
    
    print("\n\nDONE!!!")

