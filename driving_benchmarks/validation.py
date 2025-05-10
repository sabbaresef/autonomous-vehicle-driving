import argparse
import logging
import json
import os

import tensorflow as tf

import argparse
import logging

from version084.benchmark_tools import run_driving_benchmark
from version084.driving_benchmarks import Validation_new_town, Validation_new_weather, Validation_new_weather_town


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name')
    argparser.add_argument(
        '-gpu', '--gpu',
        dest='gpu',
        action='store_true',
        help='if to use gpu')
    
    ########################################################################
    argparser.add_argument('--configuration',
                        dest='configs',
                        default=os.path.dirname(os.path.realpath(__file__))+'/configs.json',
                        help='Filepath of configuration file')
    ########################################################################

    args = argparser.parse_args()
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    
    # We load configurations for the cil agent
    configuration = None
    with open(args.configs,'r') as f:
        configuration = json.load(f)

    if configuration != None:
        print("\nConfiguration Loaded")
    else:
        print("\nConfiguration Not Loaded")
        exit(0)
    
    if not args.gpu:
        print("\nGPU not used")
        tf.config.experimental.set_visible_devices([], 'GPU')
    else:
        print("\nGPU used")
    
    # We instantiate CILAgent using loaded configuration
    from version084.carla.agent.cil_agent import CILAgent
    agent = CILAgent(configuration=configuration)
    
    # We define the experiment name
    exp_name = "Validation_" + configuration["experiment_name"]
    
    # We define the folder were we locate the validation log file used to resume validation from the scenario in which it was arrived.
    validation_log = os.path.join("./validation", configuration["experiment_name"])
    
    if not os.path.exists(validation_log):
        os.makedirs(validation_log)
    
    if not os.path.exists(os.path.join(validation_log, "validation.json")):
        # If the file does not already exist it is created starting from an index 1, which smiles on the New Town scenario.
        index = 1
        data = {'val': index}
        with open(os.path.join(validation_log, "validation.json"), 'w') as f:
            json.dump(data, f)  
    else:
        # If the file already exists, the validation is resumed from the scenario in which it stopped.
        with open(os.path.join(validation_log, "validation.json"), 'r') as json_file:
            data = json.load(json_file)
            index = data['val']
    
    validation_scenarios = ['New town', 'New town and new weather'] 
    
    while index < 3:
        print(f"\n\n\nStarting validation in: {validation_scenarios[index]}")
        if index == 1:
            experiment_suite = Validation_new_town()
            city_name='Town02'
        if index == 2:
            experiment_suite = Validation_new_weather_town()
            city_name='Town02'
        
        # Now actually run the driving_benchmark
        run_driving_benchmark(agent, experiment_suite, city_name,
                            exp_name, args.continue_experiment,
                            args.host, args.port)

        data = {'val': index}
        with open(os.path.join(validation_log, "validation.json"), 'w') as f:
            json.dump(data, f)
            
        # We update the index to go to the next scenario. 
        index +=1
        
        
