import subprocess

#################################################################################################
# By specifying models architecture and town, you can perform multiple capture in sequence.
#################################################################################################
models = ["Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor", "Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor", "Resnet34_RGBD_128_batch_10h_training_5h_validation_prelu_speed_conv1_sfactor"]
town = "Town01"


# The configuration folder
configuration_folder = "/user/fsabbarese/driving_benchmarks/version084/carla/agent/cil_agent/models/configuration/"

# The index of capture to specify to avoid overwriting the previous Capture
i = 0


for model in models:
  command = "python3 capture.py --city-name='" + town + "' --continue-experiment --configuration='" + configuration_folder + model + ".json' --host dldev1.diem.unisa.it  --port 6000 -v --gpu --name='Capture_" + str(i) + "' > log_validdation_" + model + ".txt"
  print(f"Executing: {command}")
  subprocess.call(command, shell=True)
  i += 1
  