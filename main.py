from dataset_preprocessing import DatasetLoader

from models.RGBmobilenet import MobileNetV2_RGB, MobileNetV2_SingleChannel, MobileNetV2_Multichannel
from models.InceptionNetBackend import InceptionV3_RGB, InceptionV3_SingleChannel, InceptionV3_Multichannel
from models.ResNet50V2 import ResNet50V2_RGB, ResNet50V2_SingleChannel, ResNet50V2_Multichannel
from models.ResNet34 import ResNet34_RGB, ResNet34_SingleChannel, ResNet34_Multichannel

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


import tensorflow as tf
import numpy as np
import os
import glob
import json
import argparse


#######################################################
# Hide GPU from visible devices, to use CPU
# tf.config.experimental.set_visible_devices([], 'GPU')
#######################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--configuration',
                        dest='configs',
                        default=os.path.dirname(os.path.realpath(__file__))+'/configs.json',
                        help='Filepath of configuration file')

    args = parser.parse_args()
    
    # Loading the configuration file for training a model. 
    with open(args.configs,'r') as f:
        configuration = json.load(f)

    tf.print("\nConfiguration Loaded")

    experiments_dir = configuration["filepath_experiment"] + configuration["experiment_name"]

    if not os.path.exists(experiments_dir):
        os.mkdir(experiments_dir)
        os.mkdir(os.path.join(experiments_dir,"weights"))
        os.mkdir(os.path.join(experiments_dir,"logs"))
          
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.print("Device:", tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    tf.print("\nNumber of replicas:", strategy.num_replicas_in_sync)
    
    
    netName = configuration["net"] + "_" + configuration["model"]
    
    # Based on the model in the configuration file, the architecture is imported.
    if netName == "MobileNetV2_RGB" :
        cil_architecture = MobileNetV2_RGB(configuration["input_shape"],configuration)
    elif netName == "MobileNetV2_SingleChannel" :
        cil_architecture = MobileNetV2_SingleChannel(configuration["input_shape"],configuration)
    elif netName == "MobileNetV2_Multichannel" :
        cil_architecture = MobileNetV2_Multichannel(configuration["input_shape"],configuration)

    elif netName == "InceptionV3_RGB":
        cil_architecture = InceptionV3_RGB(configuration["input_shape"],configuration)
    elif netName == "InceptionV3_SingleChannel":
        cil_architecture = InceptionV3_SingleChannel(configuration["input_shape"],configuration)
    elif netName == "InceptionV3_Multichannel":
        cil_architecture = InceptionV3_Multichannel(configuration["input_shape"],configuration)
    
    elif netName == "ResNet50V2_RGB":
        cil_architecture = ResNet50V2_RGB(configuration["input_shape"],configuration)
    elif netName == "ResNet50V2_SingleChannel":
        cil_architecture = ResNet50V2_SingleChannel(configuration["input_shape"],configuration)
    elif netName == "ResNet50V2_Multichannel":
        cil_architecture = ResNet50V2_Multichannel(configuration["input_shape"],configuration)
    
    elif netName == "ResNet34_RGB":
        cil_architecture = ResNet34_RGB(configuration["input_shape"],configuration)
    elif netName == "ResNet34_SingleChannel":
        cil_architecture = ResNet34_SingleChannel(configuration["input_shape"],configuration)
    elif netName == "ResNet34_Multichannel":
        cil_architecture = ResNet34_Multichannel(configuration["input_shape"],configuration)

    # If it is necessary to load the weights of a previous training.
    if configuration["filepath_weight_preload"]:
        weight_preload_filepath = os.path.join(experiments_dir, "weights", configuration["filepath_weight_preload"])

        if os.path.exists(weight_preload_filepath):
            tf.print("Weights previous training loaded!!!")
            cil_architecture.model.load_weights(weight_preload_filepath)
    
    # Loading the training set. 
    training_files = glob.glob(configuration["training_set"]+"/*.tfrecords")
    dataset_loader_training = DatasetLoader(configuration, training=True)        
    training_generator = dataset_loader_training.get_dataset(training_files)
    tf.print("\nTraining set loaded!")

    # Loading the validation set. 
    validation_files = glob.glob(configuration["validation_set"]+"/*.tfrecords")
    dataset_loader_validation = DatasetLoader(configuration, training=False)
    validation_generator = dataset_loader_validation.get_dataset(validation_files)
    tf.print("\nValidation set loaded!\n")
    
    # We comput the number of batches in validation set
    len_validation = validation_generator.reduce(0, lambda x,_: x + 1).numpy()
    

    


    
    if configuration["training"]:
        # Here we define Tensorflow callbacks for training of the models
        callbacks = [
            ModelCheckpoint(os.path.join(experiments_dir,
                                                  "weights",
                                                  'model.{epoch:02d}-{val_loss:.2f}.h5'),
                                                   monitor='val_loss',
                                                   save_best_only=False, 
                                                   save_weights_only=True, 
                                                   save_freq='epoch'),
                                                   
            EarlyStopping(monitor='val_loss',
                                patience=configuration["earlystop_patience"],
                                mode='min',
                                min_delta=configuration["min_delta"],
                                restore_best_weights=True),
                                
            TensorBoard(log_dir=os.path.join(experiments_dir,"logs"), 
                                profile_batch=50,
                                histogram_freq=5, 
                                update_freq='epoch')
            ]

        if configuration["reduce_lr"]:
            # In case of reduce learning rate policy we define the callback            
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                                    factor=configuration["factor"], 
                                                                    patience=configuration["lr_patience"], 
                                                                    verbose=1,
                                                                    mode='min', 
                                                                    min_delta=configuration["min_delta"], 
                                                                    cooldown=0, 
                                                                    min_lr=configuration["min_lr"]))
        
        tf.print("Start training!\n")
        
        
        cil_architecture.model.fit(
            training_generator,
            steps_per_epoch=None,
            verbose=2,
            epochs=configuration["epochs"],
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=len_validation, 
            max_queue_size=50,
            initial_epoch=configuration["initial_epochs"],
            workers=100
        )
        
        # Weights are saved at the end of each training and validation epoch, 
        # as the best model may not be the one with the lowest loss.
        cil_architecture.model.save_weights(os.path.join(experiments_dir,"weights",'final_weights.hdf5'))


    else:
        test_files = glob.glob(configuration["test_set"] + "/*.tfrecords")
        dataset_loader_test = DatasetLoader(configuration, training=False)
        test_generator = dataset_loader_test.get_dataset(test_files, labeled=False)

        X = test_generator

        direction = cil_architecture.FOLLOW_LANE
        X[2] = np.array([direction])
        print(cil_architecture.predict(X))

        direction = cil_architecture.LEFT
        X[2] = np.array([direction])
        print(cil_architecture.predict(X))

        direction = cil_architecture.RIGHT
        X[2] = np.array([direction])
        print(cil_architecture.predict(X))

        direction = cil_architecture.GO_STRAIGH
        X[2] = np.array([direction])
        print(cil_architecture.predict(X))

