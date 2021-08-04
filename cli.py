import csv
import time
from glob import glob
from math import ceil
from os.path import join
from typing import List

import click
import numpy as np
import yaml

from options import HerbariumSheetsOptions


@click.command()
@click.argument('input_file')
@click.option('-o', '--output_directory', required=False, type=str, default='')
def cli(input_file, output_directory):
    start = time.time()
    options = HerbariumSheetsOptions(input_file, output_directory)

    # print(f"Analyzing file")
    # results = process(options)

    # print(f"Writing results to file")
    # write_results(options, results)

    duration = ceil((time.time() - start))
    print(f"Finished in {duration} seconds.")


if __name__ == '__main__':
    cli()
