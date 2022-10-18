#!/usr/bin/env python3

import argparse
import csv
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import transformations as tf
import numpy as np

import nut_screwing.state_monitor as sm


def parse_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--input_telemetry', required=True)
    parser.add_argument('--output_graph', required=True)
    return vars(parser.parse_args())


def euler_from_q(series_dict):
    if not len(series_dict['quaternion_w']):
        return []

    q = [series_dict['quaternion_w'][-1], series_dict['quaternion_x'][-1],
         series_dict['quaternion_y'][-1], series_dict['quaternion_z'][-1]]

    return np.degrees(tf.euler_from_quaternion(q))

def read_telemetry(input_telemetry):
    assert os.path.exists(input_telemetry)
    EXTENDED_HEADER = sm.HEADER + sm.get_3_vector('euler')
    telemetry_series = {h : [] for h in EXTENDED_HEADER}
    with open(input_telemetry, 'r') as the_file:
        has_header = csv.Sniffer().has_header(the_file.read(4096))
        the_file.seek(0)
        reader = csv.DictReader(the_file, fieldnames=sm.HEADER, delimiter=' ')
        if has_header:
            next(reader)
        for row in reader:
            for k, v in row.items():
                if float(row['time']) > 4.:
                    telemetry_series[k].append(float(v))

            ea = euler_from_q(telemetry_series)
            for k, v in zip(sm.get_3_vector('euler'), ea):
                if float(row['time']) > 4.:
                    telemetry_series[k].append(v)

    return telemetry_series


def do_graphs(input_telemetry, output_graph):
    series = read_telemetry(input_telemetry)

    fig = plt.figure(figsize=(40, 20))
    axes = fig.subplots(nrows=2, ncols=3)

    axes_00_twin = axes[0, 0].twinx()
    tx = axes_00_twin.plot(series[sm.TIME], series['translation_x'], label='translation', color='b')
    v_lx = axes[0, 0].plot(series[sm.TIME], series['velocity_l_x'], label='velocity', color='orange')
    a_lx = axes_00_twin.plot(series[sm.TIME], series['acceleration_l_x'], label='acceleration', color='r')
    f_x = axes_00_twin.plot(series[sm.TIME], series['force_x'], label='force\'', color='k')
    lines_00 = tx + v_lx + a_lx + f_x
    axes[0, 0].grid(color='k', linewidth=1, linestyle=':')
    axes[0, 0].legend(lines_00, [l.get_label() for l in lines_00])

    axes[0, 1].plot(series[sm.TIME], series['translation_y'], label='translation', color='b')
    axes[0, 1].plot(series[sm.TIME], series['velocity_l_y'], label='velocity', color='orange')
    axes[0, 1].plot(series[sm.TIME], series['acceleration_l_y'], label='acceleration', color='r')
    axes[0, 1].plot(series[sm.TIME], series['force_y'], label='force', color='k')
    axes[0, 1].grid(color='k', linewidth=1, linestyle=':')
    axes[0, 1].legend()

    axes_02_twin = axes[0, 2].twinx()
    tz = axes_02_twin.plot(series[sm.TIME], series['translation_z'], label='translation\'', color='b', linewidth=2)
    v_lz = axes[0, 2].plot(series[sm.TIME], series['velocity_l_z'], label='velocity', color='orange', linewidth=2)
    a_lz = axes[0, 2].plot(series[sm.TIME], series['acceleration_l_z'], label='acceleration', color='r', linewidth=1)
    f_z = axes[0, 2].plot(series[sm.TIME], series['force_z'], label='force', color='k', linewidth=1)
    lines_02 = tz + v_lz + a_lz + f_z
    axes[0, 2].legend(lines_02, [l.get_label() for l in lines_02])

    axes_10_twin = axes[1, 0].twinx()
    ea_x = axes_10_twin.plot(series[sm.TIME], series['euler_x'], label='euler\'', color='b', linewidth=3)
    #axes[1, 0].plot(series[sm.TIME], series['velocity_a_x'], label='velocity', color='orange')
    #axes[1, 0].plot(series[sm.TIME], series['acceleration_a_x'], label='acceleration', color='r')
    torque_x = axes[1, 0].plot(series[sm.TIME], series['torque_x'], label='force', color='k')
    axes[1, 0].grid(color='k', linewidth=1, linestyle=':')
    lines_10 = ea_x + torque_x
    axes[1, 0].legend(lines_10, [l.get_label() for l in lines_10])

    axes_11_twin = axes[1, 1].twinx()
    ea_y = axes_11_twin.plot(series[sm.TIME], series['euler_y'], label='euler\'', color='b', linewidth=3)
    #axes[1, 1].plot(series[sm.TIME], series['velocity_a_y'], label='velocity', color='orange')
    #axes[1, 1].plot(series[sm.TIME], series['acceleration_a_y'], label='acceleration', color='r')
    torque_y = axes[1, 1].plot(series[sm.TIME], series['torque_y'], label='force', color='k')
    axes[1, 1].grid(color='k', linewidth=1, linestyle=':')
    lines_11 = ea_y + torque_y
    axes[1, 1].legend(lines_11, [l.get_label() for l in lines_11])

    axes_12_twin = axes[1, 2].twinx()
    ea_z = axes[1, 2].plot(series[sm.TIME], series['euler_z'], label='euler', color='b', linewidth=3)
    v_az = axes[1, 2].plot(series[sm.TIME], series['velocity_a_z'], label='velocity', color='orange')
    a_az = axes_12_twin.plot(series[sm.TIME], series['acceleration_a_z'], label='acceleration\'', color='r', linewidth=2)
    torque_z = axes_12_twin.plot(series[sm.TIME], series['torque_z'], label='force\'', color='k', linewidth=1)
    axes[1, 2].grid(color='k', linewidth=1, linestyle=':')
    lines_12 = ea_z + v_az + a_az + torque_z
    axes[1, 2].legend(lines_12, [l.get_label() for l in lines_12])

    fig.savefig(output_graph, bbox_inches='tight')

if '__main__' == __name__:
    do_graphs(**parse_args())
