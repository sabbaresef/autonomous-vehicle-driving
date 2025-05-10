import glob
import os
import json
import numpy as np
"""
Module used to check attributes existent on data before incorporating them
to the coil dataset
"""

def orientation_vector(measurement_data):
    """Compute orientation_vector from measurements data"""
    pitch = np.deg2rad(measurement_data['rotation_pitch'])
    yaw = np.deg2rad(measurement_data['rotation_yaw'])
    orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    return orientation


def forward_speed(measurement_data):
    """Compute forward speed from measurements data"""
    vel_np = np.array([measurement_data['velocity_x'], measurement_data['velocity_y'],
                       measurement_data['velocity_z']])
    speed = np.dot(vel_np, orientation_vector(measurement_data))
    return speed


def get_speed(measurement_data):
    """ Extract the proper speed from the measurement data dict """

    # If the forward speed is not on the dataset it is because speed is zero.
    if 'playerMeasurements' in measurement_data and \
            'forwardSpeed' in measurement_data['playerMeasurements']:
        return measurement_data['playerMeasurements']['forwardSpeed']
    elif 'velocity_x' in measurement_data:  # We have a 0.9.X data here
        return forward_speed(measurement_data)
    else:  # There is no speed key, probably speed is zero.
        return 0


def check_available_measurements(episode):
    """ Try to automatically check the measurements
        The ones named 'steer' are probably the steer for the vehicle
        This needs to be made more general to avoid possible mistakes on dataset reading
    """

    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
    # Open a sample measurement
    with open(measurements_list[0]) as f:
        measurement_data = json.load(f)

    available_measurements = {}
    for meas_name in measurement_data.keys():

        # Add steer
        if 'steer' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'steer': meas_name})

        # Add Throttle
        if 'throttle' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'throttle': meas_name})

        # Add brake ( Not hand brake)
        if 'brake' in meas_name and 'noise' not in meas_name and 'hand' not in meas_name:
            available_measurements.update({'brake': meas_name})

    return available_measurements


def get_direction(measurement_data):
    """ Extract the proper direction from the measurement data dict. 
        2: follow lane
        3: turn left
        4: turn right
        5: go straight

        This value stating what the ego-vehicle would do on the next intersection. 
    """
    if 'directions' in measurement_data:
        return measurement_data['directions']
    else:  # There is no directions key
        print("There is no directions key, return 2: do not care")
        return 2


def get_brake(measurement_data):
    """ Extract the proper brake from the measurement data dict. 
    """
    if 'brake' in measurement_data:
        return measurement_data['brake']
    else:  # There is no brake key, probably brake is zero.
        print("No brake, return 0")
        return 0


def get_steer(measurement_data):
    """ Extract the proper steer from the measurement data dict. 
    """
    if 'steer' in measurement_data:
        return measurement_data['steer']
    else:  # There is no steer key, probably steer is zero.
        print("No steer, return 0")
        return 0


def get_throttle(measurement_data):
    """ Extract the proper throttle from the measurement data dict. 
    """
    if 'throttle' in measurement_data:
        return measurement_data['throttle']
    else:  # There is no throttle key, probably throttle is zero.
        print("No throttle, return 0")
        return 0
