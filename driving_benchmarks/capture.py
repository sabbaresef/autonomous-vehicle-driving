import argparse
import logging
import json
import os

import tensorflow as tf

import argparse
import logging

from version084.benchmark_tools import run_driving_benchmark
from version084.driving_benchmarks import Capture


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
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name')
    ########################################################################
    argparser.add_argument(
        '-gpu', '--gpu',
        dest='gpu',
        action='store_true',
        help='if to use gpu')
        
    argparser.add_argument('--configuration',
                        dest='configs',
                        default=os.path.dirname(os.path.realpath(__file__))+'/configs.json',
                        help='Filepath of configuration file')
    
    argparser.add_argument('--name',
                        dest='name',
                        default='Capture',
                        help='Capture name')
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

    experiment_suite = Capture(args.city_name)
    
    exp_name = args.name + "_" + configuration["experiment_name"]
    
    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                        exp_name, args.continue_experiment,
                        args.host, args.port)

