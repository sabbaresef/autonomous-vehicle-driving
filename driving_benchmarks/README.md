# DRIVING BENCGMARKS

## GUIDE

In any case, it is necessary that:
- A configuration file associated with an agent exists.
- The experiment name is specified in the configuration file under the `"experiment"` keyword.
- The weights of the model indicated in `"experiment"` name are placed in the correct folder.

---

## AGENT

When you want to use a trained agent, follow these steps:

1. Go to the folder:  
   `./version084/carla/agent/cil_agent/models/`

2. Make sure you put the configuration file in the folder `./configuration`

3. Take care to specify the `"train"` parameter inside it to `"false"`.

4. The model weights must be entered in the correct folder: `./weights`

5. Enter the correct perception backbone folder.

6. Enter the correct folder relating to the number of channels of the model.

7. Insert your model here.

---

## BENCHMARK

To run a benchmark you need to specify the following information:

- City: `Town01`, `Town02`
- Benchmark: `carla100`, `corl2017`
- A configuration file for the agent
- The host on which the simulator runs
- The port on which the simulator runs
- The GPU parameter to run the model on GPU
- `-v` (verbose) is optional

Example:

```
nohup python3 ./benchmarks_cil.py --city-name="Town02" --carla100 --continue-experiment \
  --configuration="/user/fsabbarese/driving_benchmarks/version084/carla/agent/cil_agent/models/configuration/Resnet34_RGBD_128_batch_10h_training_5h_balanced_validation_prelu_speed_conv1_sfactor.json" \
  --host dldev1.diem.unisa.it --port 6000 -v --gpu > benchmark_NoCrash_.txt &
```

You can find the results of the Benchmark in `_benchmarks_results/model_name` folder in the following files:

- `summary.csv`
- `metrics.json`

**WARNING!!!**  
The benchmark must be launched separately for the two maps, `Town01` and `Town02`.

---

## MAIN CAUSES OF FAILURE

You can get the main causes of failure by running the `metric_extraction.py` script.  
You need to specify the folder to find the `summary.csv` file of the benchmark you started.  
Running the script you can find the `metrics_extraction.json` file in the specified folder.

Example:

```python
DIR = "/user/fsabbarese/driving_benchmarks/_benchmarks_results/Resnet34_RGBD_128_batch_10h_training_5h_balanced_validation_prelu_speed_conv1_sfactor_40_epochs_CARLA100_Town02/"
```

---

## VALIDATION

To run a validation you need to specify the following information:

- A configuration file for the agent
- The host on which the simulator runs
- The port on which the simulator runs
- The GPU parameter to run the model on GPU

Example:

```
nohup python3 ./validation.py --continue-experiment \
  --configuration="/user/fsabbarese/driving_benchmarks/version084/carla/agent/cil_agent/models/configuration/Resnet34_RGBD_128_batch_10h_training_5h_balanced_validation_prelu_speed_conv1_sfactor.json" \
  --host dldev1.diem.unisa.it --port 6000 -v --gpu > Validation.txt &
```

Running the `./validation_results_extraction.py` script you can extract results and get scores that you can find in `./validation` folder.  
You must specify inside the file the agent name (`MODEL`).

Example:

```python
MODEL = "Resnet34_RGBD_128_batch_10h_training_5h_validation_linear_40_epochs"
```

---

## SEQUENTIAL VALIDATION

If you want to run different validations in sequence, you can use `execute_validations.py`.  
It is necessary to specify all the models for which it is necessary to start a validation inside the file and then start it.

Example:

```python
models = [
  "Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor",
  "Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor"
]
nohup python3 execute_validations.py > exex.txt &
```

---

## VIDEO

If you want to capture videos you can use either the `capture.py` file or the benchmark and validation ones.

You must specify the `"save_images"` parameter in the file to `"True"`:  
`./version084/benchmark_tools/benchmark_runner.py` (line 49)

Example:

```
nohup python3 ./capture.py --city-name="Town02" --continue-experiment \
  --configuration="/user/fsabbarese/driving_benchmarks/version084/carla/agent/cil_agent/models/configuration/Resnet50V2_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor.json" \
  --host dldev1.diem.unisa.it --port 6000 -v --gpu --name="Capture_41" > capture_41.txt &
```

You can also execute captures in sequence using the script `execute_captures.py`.  
Inside the file there are some fields to fill in.

---

## PRODUCE VIDEO

To produce a video you can use `./produce_videos.py`.  
Inside the file you must specify:

- `MODEL`: the agent name
- `TOWN`: the town
- `NAME`: the name of the capture, used when you run the previous script
- The success/failure cause of each episode that you can find in `summary.csv` file

Running the script, you can find the video in `./_benchmarks_results/model_name/_videos`

Example:

```python
MODEL = "Resnet50V2_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor"
TOWN = "Town02"
NAME = "Capture_27"
results = ['Completed', 'Collision with pedestrian', 'Completed', 'Completed', 'Causal Confusion', 'Collision with vehicle']
```

---

## SEMANTIC SEGMENTATION

If you want to run experiments with semantic segmentation, you need to uncomment the addition of the sensor in the benchmark or validation suite.  
This is commented to reduce the flow of data sent to the client side.

Example:

```python
#conditions.add_sensor(camera2)
```

---

## DRIVING EXAMPLE

To launch the same benchmarks present in the thesis you can run the following commands, possibly changing host and port of simulator.

**ResNet50V2**

```
nohup python3 ./benchmarks_cil.py --city-name="Town01" --carla100 --continue-experiment \
--configuration="./version084/carla/agent/cil_agent/models/configuration/Resnet50V2_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor.json" \
--host dldev1.diem.unisa.it --port 6000 -v --gpu > benchmark_NoCrash_town1.txt &

nohup python3 ./benchmarks_cil.py --city-name="Town02" --carla100 --continue-experiment \
--configuration="./version084/carla/agent/cil_agent/models/configuration/Resnet50V2_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor.json" \
--host dldev1.diem.unisa.it --port 6000 -v --gpu > benchmark_NoCrash_town2.txt &
```

**ResNet34**

```
nohup python3 ./benchmarks_cil.py --city-name="Town01" --carla100 --continue-experiment \
--configuration="./version084/carla/agent/cil_agent/models/configuration/Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor_best.json" \
--host dldev1.diem.unisa.it --port 6000 -v --gpu > benchmark_NoCrash_town1.txt &

nohup python3 ./benchmarks_cil.py --city-name="Town02" --carla100 --continue-experiment \
--configuration="./version084/carla/agent/cil_agent/models/configuration/Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor_best.json" \
--host dldev1.diem.unisa.it --port 6000 -v --gpu > benchmark_NoCrash_town2.txt &
```
