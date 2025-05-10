#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import os

####################################################################################
# The directory where to look for the tfrecords to validate.
DIR = "/mnt/sdc1/fsabbarese/dataset_15_weather/"
####################################################################################

def validate_dataset(filenames, reader_opts=None, remove=False):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    :param remove: (optional) if True is used to remove corrupted tfrecords. 
    """
    i = 0
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


if __name__ == '__main__':
    files = glob.glob(DIR + "/*.tfrecords")
    validate_dataset(files, reader_opts=None, remove=True)

