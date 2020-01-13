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
import csv
import logging
import os
import pickle

import numpy as np
from prettytable import PrettyTable

from run import run_commands, make_parser

__author__ = "Islam Elnabarawy"

logger = logging.getLogger(__name__)

description = "Gather run results from a directory"


def get_results(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        params = data['params']
        results = data['results']
    return params, results


def gather(args):
    input_dir = args.input_dir
    num_runs = args.num_runs

    x = PrettyTable()
    x.field_names = [
        'Dataset', 'ARI Mean', 'ARI Std',
        'nCat Mean', 'nCat Std', 'nClr Mean', 'nClr Std',
        'Time Mean', 'Time Std'
    ]

    writer = csv.writer(open(args.output_file, 'w'), dialect='excel')
    writer.writerow(x.field_names)

    for dataset in sorted(os.listdir(input_dir)):
        all_data = [get_results("{0}/{1}/{1}-{2}.bin".format(input_dir, dataset, ix + 1)) for ix in range(num_runs)]
        all_results = np.array([data[1] for data in all_data], dtype=np.float32)

        # get the algorithm parameters from the first set of results
        param_list = all_data[0][0]

        time_results = all_results[:, :, -1]
        perf_results = all_results[:, :, -2]
        ncat_results = all_results[:, :, -3]
        nclr_results = ncat_results if all_results.shape[2] < 5 else all_results[:, :, -4]

        avg_time = time_results.mean(axis=0)
        std_time = time_results.std(axis=0)
        avg_perf = perf_results.mean(axis=0)
        std_perf = perf_results.std(axis=0)
        avg_ncat = ncat_results.mean(axis=0)
        std_ncat = ncat_results.std(axis=0)
        avg_nclr = nclr_results.mean(axis=0)
        std_nclr = nclr_results.std(axis=0)

        # do a lexsort by best performance first, then smallest ncat, then smallest param values
        best_ix = np.lexsort([param_list[:, ix] for ix in range(param_list.shape[1])] + [avg_nclr, avg_ncat, -avg_perf])[0]

        row = [
            dataset,
            avg_perf[best_ix], std_perf[best_ix],
            avg_ncat[best_ix], std_ncat[best_ix],
            avg_nclr[best_ix], std_nclr[best_ix],
            avg_time[best_ix], std_time[best_ix]
        ]
        x.add_row(row)
        writer.writerow(row)

    print(x)


def setup_parser(parser):
    parser.add_argument("input_dir", type=str, help="The path to the directory to gather results from")
    parser.add_argument("output_file", type=str, help="Name of the CSV file to write the gathered results to")
    parser.add_argument("--num-runs", "-n", type=int, default=30,
                        help="The number of runs to gather results from")
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
