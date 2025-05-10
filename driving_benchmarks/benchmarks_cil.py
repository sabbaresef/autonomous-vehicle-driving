#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import logging
import json
import os

import tensorflow as tf

from version084.benchmark_tools import run_driving_benchmark
from version084.driving_benchmarks import CoRL2017, CARLA100
from version084.benchmark_tools.experiment_suites.basic_experiment_suite import BasicExperimentSuite
from version084.benchmark_tools.agent import ForwardAgent

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
        '--corl-2017',
        action='store_true',
        help='If you want to benchmark the corl-2017 instead of the Basic one'
    )
    argparser.add_argument(
        '--carla100',
        action='store_true',
        help='If you want to use the carla100 benchmark instead of the Basic one'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )
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
    

    # We instantiate an experiment suite. Basically a set of experiments
    # that are going to be evaluated on this benchmark.
    if args.corl_2017:
        experiment_suite = CoRL2017(args.city_name)
    elif args.carla100:
        experiment_suite = CARLA100(args.city_name)
    else:
        print (' WARNING: running the basic driving benchmark, to run for CoRL 2017'
               ' experiment suites, you should run'
               ' python driving_benchmark_example.py --corl-2017')
        experiment_suite = BasicExperimentSuite(args.city_name)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          configuration["experiment_name"], args.continue_experiment,
                          args.host, args.port)
