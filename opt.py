import argparse

parser = argparse.ArgumentParser(description="BCFW")

parser.add_argument('--lambda', type=float, default=1e-2)
parser.add_argument('--gap-threshold', type=float, default=0.1)
parser.add_argument('--num-passes', type=int, default=100)
parser.add_argument('--do-line-search', action='store_true')
parser.add_argument('--debug', action='store_true')
