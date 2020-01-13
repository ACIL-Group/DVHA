import argparse
import importlib
import logging
from collections import OrderedDict

from colorlog import ColoredFormatter

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)

description = "ART sweep evaluation CLI"

subcommands = OrderedDict([
    ("FA", "algorithms.fuzzy_art"),
    ("DVFA", "algorithms.dual_vigilance_fuzzy_art"),
    ("HA", "algorithms.hypersphere_art"),
    ("DVHA", "algorithms.dual_vigilance_hypersphere_art"),
    ("gather", "processing.gather_results"),
    ("gather_raw", "processing.gather_raw"),
    ("reorder", "processing.reorder_datasets"),
])


def configure_logging(verbosity, enable_colors):
    root_logger = logging.getLogger()
    console = logging.StreamHandler()

    if enable_colors:
        # create a colorized formatter
        formatter = ColoredFormatter(
            "%(log_color)s[%(filename)s] %(asctime)s %(levelname)-8s%(reset)s %(white)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "cyan,bg_red",
            },
            secondary_log_colors={},
            style="%"
        )
    else:
        # create a plain old formatter
        formatter = logging.Formatter(
            "[%(filename)s] %(asctime)s %(levelname)-8s %(message)s"
        )

    # add the formatter to the console handler, and the console handler to the root logger
    console.setFormatter(formatter)
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    root_logger.addHandler(console)

    # set logging level for root logger
    root_logger.setLevel(verbosity)


def make_parser(desc):
    """Construct and return a CLI argument parser.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--no-tracebacks", dest='tracebacks', action="store_false",
                        help="Disable full tracebacks")
    parser.add_argument("--verbosity", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Desired log level")
    parser.add_argument("--no-log-colors", dest='log_colors', action="store_false",
                        help="Disable log colors")

    return parser


def setup_parser(parser):
    setup_subcommands(parser, subcommands)


def setup_subcommands(parser, subcmds):
    # If no arguments are provided, show the usage screen
    parser.set_defaults(run=lambda x: parser.print_usage())

    # Set up subcommands for each package
    subparsers = parser.add_subparsers(title="subcommands")

    for name, path in subcmds.items():
        module = importlib.import_module(path)
        subparser = subparsers.add_parser(name, help=module.description)
        module.setup_parser(subparser)

    # The "help" command shows the help screen
    help_parser = subparsers.add_parser("help",
                                        help="Show this help screen and exit")
    help_parser.set_defaults(run=lambda x: parser.print_help())


def run_commands(args, log):
    # configure logging
    configure_logging(args.verbosity, args.log_colors)

    # print CLI arg values
    log_args(args, log)

    try:
        args.run(args)
    except Exception as e:
        if args.tracebacks:
            raise e
        log.error(str(e))
        raise SystemExit(1) from e


def log_args(args, log):
    arg_names = sorted(vars(args))
    fmt_str = '{:>' + str(max(map(len, arg_names))) + '} : {}'
    for a in arg_names:
        log.info(fmt_str.format(a, getattr(args, a)))


def main():
    """
    Entry point
    """
    # parse CLI args
    parser = make_parser(description)
    setup_parser(parser)
    args = parser.parse_args()

    run_commands(args, logger)


if __name__ == "__main__":
    main()
