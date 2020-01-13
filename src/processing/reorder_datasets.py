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
import logging
import multiprocessing
import os
import random

import numpy as np
from nuart.preprocessing import vat

import config
from run import run_commands, make_parser

__author__ = "Islam Elnabarawy"

logger = logging.getLogger(__name__)

description = "Reorder and save dataset files based on random seeds and VAT"


def reorder_data(args):
    data, seed, dataset, outdir_fmt, fname_fmt, inputs_fmt, labels_fmt = args

    rng_dir = outdir_fmt.format(method='rng', seed=seed)
    rng_inputs_fname = os.path.join(rng_dir, inputs_fmt.format(dataset))
    rng_labels_fname = os.path.join(rng_dir, labels_fmt.format(dataset))

    vat_dir = outdir_fmt.format(method='vat', seed=seed)
    vat_inputs_fname = os.path.join(vat_dir, fname_fmt.format(dataset))
    vat_labels_fname = os.path.join(vat_dir, labels_fmt.format(dataset))

    random.seed(int(seed))

    ix = list(range(data.shape[0]))
    random.shuffle(ix)
    data = data[ix, :]
    np.savetxt(rng_inputs_fname, data[:, :-1], delimiter=',', fmt='%g')
    np.savetxt(rng_labels_fname, data[:, -1], delimiter=',', fmt='%g')

    ix = vat(data[:, :-1])
    data = data[ix, :]
    np.savetxt(vat_inputs_fname, data[:, :-1], delimiter=',', fmt='%g')
    np.savetxt(vat_labels_fname, data[:, -1], delimiter=',', fmt='%g')

    return dataset, seed


def reorder(args):
    outdir_fmt = args.outdir_fmt
    input_dir = args.input_dir
    fname_fmt = args.fname_fmt
    num_seeds = args.num_seeds

    inputs_fmt = args.inputs_fname_fmt
    labels_fmt = args.labels_fname_fmt

    run_args = []

    for dataset in config.dataset_names:
        data = np.loadtxt(os.path.join(input_dir, fname_fmt.format(dataset)), delimiter=',')

        for seed in config.rand_seeds[:num_seeds]:
            os.makedirs(outdir_fmt.format(method='rng', seed=seed), exist_ok=True)
            os.makedirs(outdir_fmt.format(method='vat', seed=seed), exist_ok=True)

            run_args.append((data, seed, dataset, outdir_fmt, fname_fmt, inputs_fmt, labels_fmt))

    with multiprocessing.Pool() as pool:
        for dataset, seed in pool.imap_unordered(reorder_data, run_args):
            print("Reordered seed {} for dataset {}".format(seed, dataset), flush=True)


def setup_parser(parser):
    parser.add_argument("input_dir", type=str, help="The path to the directory to load data from")
    parser.add_argument("outdir_fmt", type=str, help="Name of the CSV file to write the gathered results to")
    parser.add_argument("--fname-fmt", type=str, default='{}.csv', help="Dataset filename format")
    parser.add_argument("--inputs-fname-fmt", type=str, default='{}.csv', help="Reordered inputs filename format")
    parser.add_argument("--labels-fname-fmt", type=str, default='{}.txt', help="Reordered labels filename format")
    parser.add_argument("--num-seeds", "-n", type=int, default=30, help="The number of random seeds to use")
    parser.set_defaults(run=reorder)


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
