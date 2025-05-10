# MULTIMODAL END-TO-END DRIVING THROUGH CONDITIONAL IMITATION LEARNING

## SETUP

Before running training or dataset operations, make sure to set up your Python environment installing packages:

```
pip install -r requirements.txt
```

## TRAINING

To launch a training:

- Create a configuration file for the agent
- Start the training running `main.py`, passing the configuration file

This is an example:

```
nohup python3 main.py \
--configuration=./configuration_ResNet34/Resnet34_RGBD_128_batch_10h_training_5h_balanced_validation_linear.json > train.txt &
```

Alternatively, you can use a `.sh` file like the ones in this folder.

---

## DATASET

To create a dataset in the `tfrecords` format you have to follow the following steps:

1. In the `dataset_loader.py` file you must specify the characteristics of the dataset in the dictionary `configuration`. You need to specify the batch size, if you want balancing or shuffling. In addition, the output directory and the number of hours must be defined.

2. Run the `dataset_loader.py` file. Example:
   ```
   nohup python3 dataset_loader.py > load_balanced.txt &
   ```

3. At the end of execution you will find a `.npy` file in the specified output directory.

4. In the `dataset_preprocessing.py` file set the `convert` flag to `True`.

5. Specify the folder where to find the `.npy` file created in the `DIR` constant.

6. Create the dataset converter passing the directory in `DIR` and the name of the `.npy` file.

7. Call the `.convert_dataset()` function passing to it the directory in which to write the `tfrecords` and the maximum number of samples for `tfrecord`.

8. You can run the file `dataset_preprocessing.py`. Example:
   ```
   nohup python3 dataset_preprocessing.py > preproc.txt &
   ```

9. You can see the `tfrecord` in the output directory.

10. You can validate and move `tfrecords`, to fill training set and validation set. To make this, you can use scripts in **UTILS** guide below.

---

## UTILS

In the `./utils` folder you can find some useful scripts.

- `dataset_analisys.py`: This script allows you to analyze the dataset that is passed. Provides sample statistics for direction, weather, and steer. It also saves the results on file and creates graphs.

- `move_tfrecords.py`: Can be used to move samples from one folder to another.

- `validate_dataset.py`: If some `tf records` are corrupt, this script allows you to locate and delete them.

- `split_dataset.py`: Splits the dataset into two partitions based on a splitting value.

- `validate_and_split_dataset.py`: Allows you to validate and split the dataset as already indicated.

---

## MODELS

In the `./models` folder you can find the CIL architecture and the different CILRS implementations with various perception backbones. Please note that `ResNet50V2` and `ResNet34` have been added from the baseline provided.

You can also find the application of the `Tanh-PReLU` configuration described in the thesis.

You can easily add new perception backbones by extending the `CIL` class in `model.py`.

**WARNING!!!**  
`ResNet34`: This perception backbone is not present in the TensorFlow packages. To import it you can refer to the guide on the following page:  
[https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models)

---

## AUGMENTATION

Two scripts are provided for data augmentation:  
- `./Automold.py`  
- `./augmentation.py`

The first defines the transformations described in the thesis.  
In the second, classical transformations are implemented.

You can find Automold library on the following page:  
[https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library)
