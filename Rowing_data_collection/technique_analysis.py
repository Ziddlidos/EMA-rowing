'''
This script was made for the rowing data collection work.
Its purpose is to analyze diferent datasets as to how long the rower activated each rowing phase.
By doing that it is possible to infer if they kept a consistent rowing pattern.
Author: Lucas Fonseca
Contact: lucasafonseca@lara.unb.br
Date: Feb 25th 2019
'''

import matplotlib.pyplot as plt
import pickle
from data_classification import *
import numpy as np

filename = 'Estevao_rowing.out'

plt.rcParams['svg.fonttype'] = 'none'

data = {}

# Load data
with open(filename, 'rb') as f:
    try:
        while True:
            print('Loading...')
            data.update({pickle.load(f): pickle.load(f)})

    except EOFError:
        print('Loading complete')

var_names = []
for k, v in data.items():
    var_names.append(k)

print('Variables loaded: ', var_names)

# Assign variables
[buttons_timestamp, buttons_values] = [data['buttons_timestamp'], data['buttons_values']]
imus = data['imus']
[emg_1_timestamp, emg_1_values] = [data['emg_1_timestamp'], data['emg_1_values']]
[emg_2_timestamp, emg_2_values] = [data['emg_2_timestamp'], data['emg_2_values']]

[low, zero, up] = classify_by_buttons(buttons_timestamp, buttons_values, imus[2].timestamp, imus[2].euler_z)

training_lower_time_table = [200, 300, 400, 500, 600]
training_time = 75
testing_time = 25

plt.figure()
i = 0
for starting_time in training_lower_time_table:
    i += 1

    print('Time frame: {}-{}s'.format(starting_time, starting_time + training_time + testing_time))

    time_in_low = []
    time_in_zero = []
    time_in_up = []
    timestamp_in_low = []
    timestamp_in_zero = []
    timestamp_in_up = []

    for packet in low:
        if starting_time < packet.timestamp[0] < (starting_time + training_time + testing_time):
            time_in_low.append(packet.timestamp[-1] - packet.timestamp[0])
            timestamp_in_low.append([0])
    for packet in zero:
        if starting_time < packet.timestamp[0] < (starting_time + training_time + testing_time):
            time_in_zero.append(packet.timestamp[-1] - packet.timestamp[0])
            timestamp_in_zero.append([0])
    for packet in up:
        if starting_time < packet.timestamp[0] < (starting_time + training_time + testing_time):
            time_in_up.append(packet.timestamp[-1] - packet.timestamp[0])
            timestamp_in_up.append([0])

    # print('Total time in flexion: {}s'.format(sum(time_in_low)))
    # print('Total time in stop: {}s'.format(sum(time_in_zero)))
    # print('Total time in extension: {}s'.format(sum(time_in_up)))

    plt.subplot(1, len(training_lower_time_table), i)
    plt.plot(time_in_low, label='Flexion')
    plt.plot(time_in_zero, label='Stop')
    plt.plot(time_in_up, label='Extension')
    plt.title('Time frame: {}-{}s'.format(starting_time, starting_time + training_time + testing_time))
    # plt.xlabel('Flexion - mean = {}s, var = {}s\nStop - mean = {}s, var = {}\nExtension - mean = {}s, var ={}s'
    #            .format(round(np.mean(time_in_low), 2), round(np.var(time_in_low), 2),
    #                    round(np.mean(time_in_zero), 2), round(np.var(time_in_zero), 2),
    #                    round(np.mean(time_in_up), 2), round(np.var(time_in_up), 2)))


    if i == 1:
        plt.ylabel('Period in each phase [s]')
        plt.legend(ncol=3)
    elif i == 3:
        plt.xlabel('Stroke')

plt.show()
