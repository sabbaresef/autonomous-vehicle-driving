from __future__ import print_function

from ..benchmark_tools.experiment import Experiment
from ..carla.sensor import Camera
from ..carla.settings import CarlaSettings
from ..benchmark_tools.experiment_suites.experiment_suite import ExperimentSuite


class Validation_new_weather_town(ExperimentSuite):

    def __init__(self):
        super(Validation_new_weather_town, self).__init__('Town02')

    @property
    def train_weathers(self):
        return []
        
    @property
    def test_weathers(self):
        return [14]
        
    @property
    def collision_as_failure(self):
        return True
    @property
    def traffic_light_as_failure(self):
        return False
    
    def calculate_time_out(self, path_distance):
        """
        Function to return the timeout ,in milliseconds,
        that is calculated based on distance to goal.
        This timeout is increased since stop for traffic lights is expected.
        """
        return ((path_distance / 1000.0) / 5.0) * 3600.0 + 20.0
        

    def _poses(self):

        return [[[19, 66], [79, 14], [19, 57], [23, 1],
                [53, 76], [42, 13], [31, 71], [33, 5],
                [54, 30], [10, 61], [66, 3], [27, 12],
                [79, 19], [2, 29], [16, 14], [5, 57],
                [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                [51, 81], [77, 68], [56, 65], [43, 54]]]


    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set sensors
        camera0 = Camera('CentralRGB')
        camera0.set_image_size(800, 600)
        camera0.set(FOV=100)
        camera0.set_position(2.0, 0.0, 1.4)
        camera0.set_rotation(-15.0, 0, 0)

        camera1 = Camera('CentralDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        camera1.set(FOV=100)
        camera1.set_position(2.0, 0.0, 1.4)
        camera1.set_rotation(-15.0, 0, 0)

        camera2 = Camera('CentralSemanticSeg', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(800, 600)
        camera2.set(FOV=100)
        camera2.set_position(2.0, 0.0, 1.4)
        camera2.set_rotation(-15.0, 0, 0)

        poses_tasks = self._poses()
        vehicles_tasks = [15]
        pedestrians_tasks = [50]

        experiments_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                
                # Add all the cameras that were set for this experiments
                conditions.add_sensor(camera0)
                conditions.add_sensor(camera1)
                #conditions.add_sensor(camera2)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector
