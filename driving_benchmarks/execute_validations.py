import subprocess

#############################################################################
# By specifying models, you can perform multiple validations in sequence.
#############################################################################

models = ["Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor", "Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_augmentation_strong_sfactor"]

# The configuration folder
configuration_folder = "/user/fsabbarese/driving_benchmarks/version084/carla/agent/cil_agent/models/configuration/"

for model in models:
  command = "python3 validation.py --continue-experiment --configuration='" + configuration_folder + model + ".json' --host dldev1.diem.unisa.it  --port 6000 -v --gpu > log_validdation_" + model + ".txt"
  print(f"Executing: {command}")
  subprocess.call(command, shell=True)
  
  
