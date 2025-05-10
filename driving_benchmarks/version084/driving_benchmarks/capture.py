
from __future__ import print_function

from ..benchmark_tools.experiment import Experiment
from ..carla.sensor import Camera
from ..carla.settings import CarlaSettings
from ..benchmark_tools.experiment_suites.experiment_suite import ExperimentSuite



class Capture(ExperimentSuite):

    @property
    def train_weathers(self):
        return [1, 3, 6, 8]

    @property
    def test_weathers(self):
        return [10, 14]

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

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks

        """

        def _poses_navigation():
            return [[27, 130]]

        return [_poses_navigation()]

    def _poses_town02(self):

        def _poses_navigation():
            #return [[24, 54]]
            return [[41, 57], [24, 54], [20, 81]]
        return [_poses_navigation()]

    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set sensors
        camera0 = Camera('CentralRGB')
        camera0.set_image_size(800, 600)
        #camera0.set_image_size(200, 88)
        camera0.set(FOV=100)
        camera0.set_position(2.0, 0.0, 1.4)
        camera0.set_rotation(-15.0, 0, 0)

        camera1 = Camera('CentralDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        #camera1.set_image_size(200, 88)
        camera1.set(FOV=100)
        camera1.set_position(2.0, 0.0, 1.4)
        camera1.set_rotation(-15.0, 0, 0)

        camera2 = Camera('CentralSemanticSeg', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(800, 600)
        #camera2.set_image_size(200, 88)
        camera2.set(FOV=100)
        camera2.set_position(2.0, 0.0, 1.4)
        camera2.set_rotation(-15.0, 0, 0)

        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [150]
            pedestrians_tasks = [300]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [50]
            pedestrians_tasks = [85]

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

                conditions.set(DisableTwoWheeledVehicles=True)
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera0)
                conditions.add_sensor(camera1)
                #conditions.add_sensor(camera2)
                
                conditions.randomize_seeds()
                
                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector
