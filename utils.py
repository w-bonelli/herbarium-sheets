import csv
from glob import glob
from os.path import join
from typing import List

import yaml

from options import AnalysisOptions
from results import AnalysisResult


def list_images(path: str, filetypes: List[str]):
    files = []
    for filetype in filetypes:
        files = files + sorted(glob(join(path, f"*.{filetype}")))
    return files


def write_results(options: AnalysisOptions, results: List[AnalysisResult]):
    # YAML
    with open(join(options.output_directory, f"{options.input_stem}.results.yml"), 'w') as file:
        yaml.dump({'features': results}, file, default_flow_style=False)

    # CSV
    with open(join(options.output_directory, f"{options.input_stem}.results.csv"), 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if len(results) != 0:
            writer.writerow(list(results[0].keys()))
        for result in results:
            writer.writerow(list(result.values()))
