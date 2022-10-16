#!/usr/bin/env python3

import argparse
import csv
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nut_screwing.state_monitor as sm


def parse_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--input_telemetry', required=True)
    parser.add_argument('--output_graph', required=True)
    return vars(parser.parse_args())


def read_telemetry(input_telemetry):
    assert os.path.exists(input_telemetry)
    telemetry_series = {h : [] for h in sm.HEADER}
    with open(input_telemetry, 'r') as the_file:
        has_header = csv.Sniffer().has_header(the_file.read(2048))
        the_file.seek(0)
        reader = csv.DictReader(the_file, fieldnames=sm.HEADER, delimiter=' ')
        if has_header:
            next(reader)
        for row in reader:
            for k, v in row.items():
                telemetry_series[k].append(float(v))

    return telemetry_series


def do_graphs(input_telemetry, output_graph):
    series = read_telemetry(input_telemetry)

    fig = plt.figure(figsize=(40, 20))
    axes = fig.subplots(nrows=2, ncols=3)

    axes[0, 0].plot(series[sm.TIME], series['translation_x'], label='translation', color='b')
    axes[0, 0].plot(series[sm.TIME], series['velocity_l_x'], label='velocity', color='orange')
    axes[0, 0].plot(series[sm.TIME], series['acceleration_l_x'], label='acceleration', color='r')
    axes[0, 0].plot(series[sm.TIME], series['force_x'], label='force', color='k')
    axes[0, 0].grid(color='k', linewidth=1, linestyle='-')
    axes[0, 0].legend()

    axes[0, 1].plot(series[sm.TIME], series['translation_y'], label='translation', color='b')
    axes[0, 1].plot(series[sm.TIME], series['velocity_l_y'], label='velocity', color='orange')
    axes[0, 1].plot(series[sm.TIME], series['acceleration_l_y'], label='acceleration', color='r')
    axes[0, 1].plot(series[sm.TIME], series['force_y'], label='force', color='k')
    axes[0, 1].grid(color='k', linewidth=1, linestyle='-')
    axes[0, 1].legend()

    axes[0, 2].plot(series[sm.TIME], series['translation_z'], label='translation', color='b')
    axes[0, 2].plot(series[sm.TIME], series['velocity_l_z'], label='velocity', color='orange')
    axes[0, 2].plot(series[sm.TIME], series['acceleration_l_z'], label='acceleration', color='r')
    axes[0, 2].plot(series[sm.TIME], series['force_z'], label='force', color='k')
    axes[0, 2].grid(color='k', linewidth=1, linestyle='-')
    axes[0, 2].legend()

    axes[1, 0].plot(series[sm.TIME], series['velocity_a_x'], label='velocity', color='orange')
    axes[1, 0].plot(series[sm.TIME], series['acceleration_a_x'], label='acceleration', color='r')
    axes[1, 0].plot(series[sm.TIME], series['torque_x'], label='force', color='k')
    axes[1, 0].grid(color='k', linewidth=1, linestyle='-')
    axes[1, 0].legend()

    axes[1, 1].plot(series[sm.TIME], series['velocity_a_y'], label='velocity', color='orange')
    axes[1, 1].plot(series[sm.TIME], series['acceleration_a_y'], label='acceleration', color='r')
    axes[1, 1].plot(series[sm.TIME], series['torque_y'], label='force', color='k')
    axes[1, 1].grid(color='k', linewidth=1, linestyle='-')
    axes[1, 1].legend()

    axes[1, 2].plot(series[sm.TIME], series['velocity_a_z'], label='velocity', color='orange')
    axes[1, 2].plot(series[sm.TIME], series['acceleration_a_z'], label='acceleration', color='r')
    axes[1, 2].plot(series[sm.TIME], series['torque_z'], label='force', color='k')
    axes[1, 2].grid(color='k', linewidth=1, linestyle='-')
    axes[1, 2].legend()

    fig.savefig(output_graph, bbox_inches='tight')

if '__main__' == __name__:
    do_graphs(**parse_args())
