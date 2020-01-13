"""
   Copyright 2019 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import itertools
import logging
import multiprocessing
import os
import pickle
import random
import time
import timeit
from math import sqrt

import numpy as np
from nuart.clustering import DualVigilanceHypersphereART
from nuart.preprocessing import vat
from prettytable import PrettyTable
from sklearn.metrics import adjusted_rand_score

import config
from run import run_commands, make_parser

__author__ = "Islam Elnabarawy"

logger = logging.getLogger(__name__)

description = "Run the Dual Vigilance Hypersphere ART clustering algorithm"


def run_single(inputs, targets, rho_ub, rho_lb, r_bar, alpha, beta, max_epochs):
    # run DualVigilanceFuzzyART and get labels
    dvha = DualVigilanceHypersphereART(rho_ub, rho_lb, r_bar, alpha, beta, max_epochs=max_epochs, shuffle=False)

    timer = timeit.default_timer()
    labels = dvha.fit(inputs)
    elapsed = timeit.default_timer() - timer

    # calculate adjusted rand index performance
    performance = adjusted_rand_score(targets, labels)

    # return results
    return dvha.iterations, dvha.num_clusters, dvha.num_categories, performance, elapsed


def run_sweep(inputs, targets, params, max_epochs, enable_mp, num_processes):
    # get an iterator with all the different parameter combinations
    run_args = [(inputs, targets, *combs, max_epochs) for combs in params]

    # use the built-in map function by default
    map_fn = itertools.starmap

    if enable_mp:
        # use the multiprocessing Pool.map function if possible
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        try:
            pool = multiprocessing.Pool(num_processes)
            map_fn = pool.starmap
        except Exception:
            logger.exception("Failed to initialize multiprocessing pool.")

    return list(map_fn(run_single, run_args))


def evaluate(args):
    # load the dataset
    dataset = np.loadtxt('data/csv/{}.csv'.format(args.dataset), delimiter=',')
    inputs, targets = dataset[:, :-1], dataset[:, -1]

    logger.info("Loaded dataset {} with {} attributes, {} samples, and {} labels.".format(
        args.dataset, inputs.shape[1], inputs.shape[0], np.unique(targets).size
    ))

    random_seed = args.random_seed if args.random_seed is not None \
        else int(time.time())

    logger.info('Random number generator seed: {}'.format(random_seed))

    results = []
    done_params = []

    if args.restore_file is not None and not os.path.isfile(args.restore_file):
        logger.warning('Restore file not found at the specified path: {}. Starting fresh!'.format(args.restore_file))
        args.restore_file = None

    if args.restore_file is not None:
        # load params and results from restore file
        with open(args.restore_file, 'rb') as f:
            restore_data = pickle.load(f)

        done_params = restore_data['done_params']
        params = restore_data['params']
        results = restore_data['results']

        # if the random seed was not specified, keep the previous one
        if args.random_seed is None:
            random_seed = restore_data['random_seed']

        logger.info('Restored checkpoint from file: {}'.format(args.restore_file))
        logger.info('Found {} completed results.'.format(len(results)))
        if params:
            logger.info('Working on {} remaining out of {} total combinations.'.format(
                len(params), len(params) + len(done_params)))
        else:
            logger.info('No remaining combinations to process in checkpoint file.')

    else:
        rho_ub = np.array([args.rho_ub]) if args.rho_ub is not None \
            else np.linspace(args.rho_ub_min, args.rho_ub_max, num=args.rho_ub_count)

        rho_lb = np.array([args.rho_lb]) if args.rho_lb is not None \
            else np.linspace(args.rho_lb_min, args.rho_lb_max, num=args.rho_lb_count)

        r_max = sqrt(inputs.shape[1]) / 2
        r_bar = r_max * (np.array([args.r_bar]) if args.r_bar is not None
                         else np.linspace(args.r_bar_min, args.r_bar_max, num=args.r_bar_count))

        alpha = np.array([args.alpha]) if args.alpha is not None \
            else np.linspace(args.alpha_min, args.alpha_max, num=args.alpha_count)

        beta = np.array([args.beta]) if args.beta is not None \
            else np.linspace(args.beta_min, args.beta_max, num=args.beta_count)

        rho_values = [(ub, lb) for (ub, lb) in itertools.product(rho_ub, rho_lb) if ub >= lb]
        params = [(v, w, x, y, z) for (v, w), x, y, z in itertools.product(rho_values, r_bar, alpha, beta)]

        logger.info('Running {} combinations of:'.format(len(params)))
        logger.info(' - {} rho_ub values'.format(len(rho_ub)))
        logger.info(' - {} rho_lb values'.format(len(rho_lb)))
        logger.info(' - {} r_bar values'.format(len(r_bar)))
        logger.info(' - {} alpha values'.format(len(alpha)))
        logger.info(' - {} beta values'.format(len(beta)))

    # seed the random number generator
    random.seed(random_seed)

    # shuffle the dataset once at the beginning
    rand_ix = list(range(dataset.shape[0]))
    random.shuffle(rand_ix)
    inputs = inputs[rand_ix, :]
    targets = targets[rand_ix]

    if args.apply_vat:
        # apply vat ordering
        vat_ix = vat(inputs)
        inputs = inputs[vat_ix, :]
        targets = targets[vat_ix]

    if args.checkpoint_file is not None:
        batch_size = args.checkpoint_freq
        num_threads = args.num_processes if args.num_processes is not None else multiprocessing.cpu_count()
        if args.enable_mp and batch_size < 10 * num_threads:
            logger.warning('The specified checkpoint frequency is too low for the current multiprocessing settings. '
                           'It should be 10 * {} or higher to take advantage of multiprocessing.'.format(num_threads))

        while params:
            batch_params, params = params[:batch_size], params[batch_size:]
            results.extend(run_sweep(inputs, targets, batch_params, args.max_epochs,
                                     args.enable_mp, args.num_processes))
            done_params.extend(batch_params)

            with open(args.checkpoint_file, 'wb') as f:
                pickle.dump({
                    'params': params,
                    'results': results,
                    'done_params': done_params,
                    'random_seed': random_seed
                }, f)

            logger.debug('Saved checkpoint after {} out of {} combinations.'.format(
                len(done_params), len(params) + len(done_params)))
    else:
        results.extend(run_sweep(inputs, targets, params, args.max_epochs, args.enable_mp, args.num_processes))
        done_params.extend(params)

    # lower the storage class on the results to float32
    results = np.array(results, dtype=np.float32)
    params = np.array(done_params, dtype=np.float32)

    x = PrettyTable()
    x.field_names = [
        'Idx', 'Rho_ub', 'Rho_lb', 'R_bar', 'Alpha', 'Beta',
        '# Iter', '# Clusters', '# Categories', 'ARI Performance', 'Time'
    ]

    for ix, val in enumerate(results):
        x.add_row([ix + 1] + list(params[ix]) + list(val))

    x.add_row(['-' * len(f) for f in x.field_names])

    avg = results.mean(axis=0)
    x.add_row(['Avg'] + ['-'] * 5 + list(avg))

    best_ix = results[:, 3].argmax()
    best_row = results[best_ix]
    best_params = params[best_ix]
    x.add_row(['Best'] + list(best_params) + list(best_row))

    print(x)

    if args.output_file is not None:
        with open(args.output_file, 'wb') as f:
            pickle.dump({
                'params': params,
                'results': results,
                'average': avg,
                'best_params': best_params,
                'best_values': best_row,
                'args': args,
                'random_seed': random_seed
            }, f)

    logger.info("Done.")


def setup_parser(parser):
    parser.add_argument("dataset", choices=config.dataset_names, help='The name of the dataset to use')
    parser.add_argument("--rho_ub", type=float, default=None,
                        help="Rho_ub parameter value to use instead of sweeping for it")
    parser.add_argument("--rho_ub-min", type=float, default=0)
    parser.add_argument("--rho_ub-max", type=float, default=1)
    parser.add_argument("--rho_ub-count", type=int, default=101)
    parser.add_argument("--rho_lb", type=float, default=None,
                        help="Rho_lb parameter value to use instead of sweeping for it")
    parser.add_argument("--rho_lb-min", type=float, default=0)
    parser.add_argument("--rho_lb-max", type=float, default=1)
    parser.add_argument("--rho_lb-count", type=int, default=101)
    parser.add_argument("--r_bar", type=float, default=None,
                        help="R_bar parameter value to use instead of sweeping for it")
    parser.add_argument("--r_bar-min", type=float, default=1)
    parser.add_argument("--r_bar-max", type=float, default=2)
    parser.add_argument("--r_bar-count", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Alpha parameter value to use instead of sweeping for it")
    parser.add_argument("--alpha-min", type=float, default=0)
    parser.add_argument("--alpha-max", type=float, default=1)
    parser.add_argument("--alpha-count", type=int, default=11)
    parser.add_argument("--beta", type=float, default=None,
                        help="Beta parameter value to use instead of sweeping for it")
    parser.add_argument("--beta-min", type=float, default=0)
    parser.add_argument("--beta-max", type=float, default=1)
    parser.add_argument("--beta-count", type=int, default=11)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--apply-vat", action='store_true', dest='apply_vat', default=False)
    parser.add_argument("--disable-mp", action='store_false', dest='enable_mp', default=True)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None,
                        help="Name of the file to pickle the results to")
    parser.add_argument("--restore-file", type=str, default=None,
                        help="Name of the file to restore a checkpoint from")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                        help="Name of the file to write checkpoints to")
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                        help="Frequency of writes to checkpoint file")
    parser.set_defaults(run=evaluate)


def main():
    """
    Entry point
    """
    # parse CLI args
    parser = make_parser(description)
    setup_parser(parser)
    args = parser.parse_args()

    run_commands(args, logger)


if __name__ == '__main__':
    main()
