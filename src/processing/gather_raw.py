"""
   Copyright 2018 Islam Elnabarawy

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
import csv
import itertools as it
import logging
import multiprocessing as mp
import os

import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import adjusted_rand_score

import config
from run import run_commands, make_parser

__author__ = "Islam Elnabarawy"

logger = logging.getLogger(__name__)

description = "Gather run results from a directory using raw TopoART csv files instead of bin files"


def process_row(row, targets):
    row_data = np.array(row, dtype=np.float32)
    ari = adjusted_rand_score(targets, row_data[7:])
    return row_data[:4], np.concatenate((row_data[5:7], [ari]))


def get_results(inputs_fname, targets_fname, module_ix, map_fn=it.starmap):
    targets = np.loadtxt(targets_fname, delimiter=',')
    with open(inputs_fname, 'r') as f:
        reader = csv.reader(f)
        params, results = zip(*map_fn(
            process_row, zip(filter(lambda row: int(row[4]) == module_ix, reader), it.repeat(targets))
        ))
    return np.array(params), np.array(results)


def gather(args):
    input_dir = args.input_dir
    targets_dir = args.targets_dir
    num_runs = args.num_runs
    module_ix = args.topo_module - 1  # convert to module ID to 0-based indexing

    x = PrettyTable()
    x.field_names = [
        'Dataset', 'ARI Mean', 'ARI Std',
        'nCat Mean', 'nCat Std', 'nClr Mean', 'nClr Std'
    ]

    writer = csv.writer(open(args.output_file, 'w'), dialect='excel')
    writer.writerow(x.field_names)

    pool = mp.Pool()

    for dataset in sorted(os.listdir(input_dir)):
        logger.info("Gathering dataset: %s", dataset)

        input_fnames = ["{0}/{1}/{1}-{2}.csv".format(input_dir, dataset, ix + 1) for ix in range(num_runs)]
        target_fnames = ["{0}/{1}/{2}.txt".format(targets_dir, config.rand_seeds[ix], dataset) for ix in
                         range(num_runs)]
        all_data = list(pool.starmap(
            get_results, zip(input_fnames, target_fnames, it.repeat(module_ix))
        ))
        all_results = np.array([data[1] for data in all_data], dtype=np.float32)

        # get the algorithm parameters from the first set of results
        param_list = all_data[0][0]

        ncat_results = all_results[:, :, 0]
        nclr_results = all_results[:, :, 1]
        perf_results = all_results[:, :, 2]

        avg_perf = perf_results.mean(axis=0)
        std_perf = perf_results.std(axis=0)
        avg_ncat = ncat_results.mean(axis=0)
        std_ncat = ncat_results.std(axis=0)
        avg_nclr = nclr_results.mean(axis=0)
        std_nclr = nclr_results.std(axis=0)

        # do a lexsort by best performance first, then smallest ncat, then smallest param values
        best_ix = np.lexsort(
            [param_list[:, ix] for ix in range(param_list.shape[1])] +
            [avg_nclr, avg_ncat, -avg_perf]
        )[0]

        row = [
            dataset,
            avg_perf[best_ix], std_perf[best_ix],
            avg_ncat[best_ix], std_ncat[best_ix],
            avg_nclr[best_ix], std_nclr[best_ix],
        ]
        x.add_row(row)
        writer.writerow(row)

    print(x)

    pool.close()


def setup_parser(parser):
    parser.add_argument("input_dir", type=str, help="The path to the directory to gather results from")
    parser.add_argument("targets_dir", type=str, help="The path to the directory to get target labels from")
    parser.add_argument("output_file", type=str, help="Name of the CSV file to write the gathered results to")
    parser.add_argument("--num-runs", "-n", type=int, default=30,
                        help="The number of runs to gather results from")
    parser.add_argument("--topo-module", "-m", type=int, default=2,
                        help="Which TopoART module to gather results for (starting from 1)")
    parser.set_defaults(run=gather)


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
