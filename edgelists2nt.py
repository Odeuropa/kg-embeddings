import argparse
import os
import csv
from tqdm import tqdm
from urllib.parse import unquote

DATA_FOLDER = './data'


def run(data_folder):
    nt = open('all.nt', 'w')
    for x in tqdm(os.listdir(data_folder)):
        if not x.endswith('.csv'):
            continue
        with open(os.path.join(data_folder, x), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            first = True
            for s, o in csv_reader:
                if first:
                    first = False
                    continue
                pred = x # unquote(x)
                nt.write('<%s> <%s> <%s> .\n' % (s, pred, o))
    nt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert Edgelist folder to nt file')
    parser.add_argument('--data', '-d', default=DATA_FOLDER)

    args = parser.parse_args()
    run(args.data)
