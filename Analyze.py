
from Analysis import analyse, mkdir
import logging
import importlib
import sys
import os
import argparse


def get_parser():

    parser = argparse.ArgumentParser(description="Analyze Seed Spacing")
    parser.add_argument('-d', '--detector_file', help="Seed Detector Location")
    parser.add_argument('-o', '--output_dir', help="Seed Detector Location")
    return parser

def get_detector(detector_file):

    try:
        sys.path.append(os.path.dirname(detector_file))
        detector_current = importlib.import_module(os.path.basename(detector_file).split(".")[0])
        detector = detector_current.detector
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(detector_file))
    return detector



if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    exps = [ "2-3" + str(i) for i in range(4, 8) ]
    results_path = args.output_dir

    mkdir(results_path)

    logging.basicConfig(filename=os.path.join(results_path, "analysis.log"),
                        format='%(asctime)s %(name)-12s %(levelname)-8s  %(message)s',
                        level=logging.DEBUG, filemode="w")

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(message)s')

    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    detector = get_detector(detector_file=args.detector_file)


    for exp in exps:
        logging.warning("Running On the DataSet " + exp)
        analyse(exp, detector=detector, results_path=results_path)
