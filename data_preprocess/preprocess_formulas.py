#!/usr/bin/env python
# tokenize latex formulas
import sys
import os
import argparse
import logging
import subprocess
import shutil
import re


def process_args(args):
    parser = argparse.ArgumentParser(description='Preprocess (tokenize or normalize) latex formulas')

    parser.add_argument('--mode', dest='mode',
                        choices=['tokenize', 'normalize'], required=True,
                        help=('Tokenize (split to tokens seperated by space) or normalize (further translate to an equivalent standard form).'
                              ))
    parser.add_argument('--input-file', dest='input_file',
                        type=str, required=True,
                        help=('Input file containing latex formulas. One formula per line.'
                              ))
    parser.add_argument('--output-file', dest='output_file',
                        type=str, required=True,
                        help=('Output file.'
                              ))
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'
                              ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt'
                              ))
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)

    input_file = parameters.input_file
    output_file = parameters.output_file

    assert os.path.exists(input_file), input_file
    temp_file = input_file + '.tmp'
    with open(temp_file, 'w') as fout:
        with open(input_file) as fin:
            for line in fin:
                fout.write(next_prepocess(line).strip() + '\n')

    cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s" % (temp_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        logging.error('FAILED: %s' % cmd)
    os.remove(temp_file)

    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as fout:
        with open(output_file) as fin:
            for line in fin:
                fout.write(line.replace('\r', ' ').replace('&gt;', '>').replace('&lt;', '<').strip() + '\n')  # delete \r

    cmd = "cat %s | node preprocess_latex.js %s > %s " % (temp_file, parameters.mode, output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        logging.error('FAILED: %s' % cmd)
    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)
    with open(temp_file) as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    tokens_out.append(token)
                newline = ' '.join(tokens_out)+'\n'
                newline = next_prepocess(newline)
                fout.write(newline)
    os.remove(temp_file)


def next_prepocess(line):
    newline = re.sub(r" ~ ~ ~ ", ' ', line)
    newline = re.sub(r"&gt;", '>', newline)
    newline = re.sub(r"&lt;", '>', newline)
    newline = re.sub(r" \\ \\ \\ \\ ", ' ', newline)
    newline = re.sub(r" \\; \\; \\; ", ' ', newline)
    newline = re.sub(r" f r a c ", r' \\frac ', newline)
    newline = re.sub(r"\\operatorname \{ ([a-z]+) ([a-z]+) \}", r'\\\1\2', newline)
    newline = re.sub(r"\\operatorname \{ ([a-z]+) ([a-z]+) ([a-z]+) \}", r'\\\1\2\3', newline)
    newline = re.sub(r"\\operatorname \{ ([a-z]+) ([a-z]+) ([a-z]+) ([a-z]+) \}", r'\\\1\2\3\4', newline)
    newline = re.sub(r"\\operatorname\* \{ ([a-z]+) ([a-z]+) \}", r'\\\1\2', newline)
    newline = re.sub(r"\\operatorname\* \{ ([a-z]+) ([a-z]+) ([a-z]+) \}", r'\\\1\2\3', newline)
    newline = re.sub(r"\\operatorname\* \{ ([a-z]+) ([a-z]+) ([a-z]+) ([a-z]+) \}", r'\\\1\2\3\4', newline)
    newline = re.sub(r" { \\kern 1 p t } ", r' ', newline)
    newline = re.sub(r" \\hspace \{ [ 0-9\-a-z]+ p t \} ", r' ', newline)
    newline = re.sub(r" ule \{ [ 0-9a-z.-]+ \} \{ [ 0-9a-z.-]+ \} ", r' ', newline)
    return newline


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
