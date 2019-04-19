'''
This script was made for the rowing data collection work.
Its purpose is to
- load binary data from a file
- classify IMU data according to button data
- Train an LDA system to classify IMU data
- Evaluate performance
It writes the results in a log file
Several parameters are editable throughout the script
Author: Lucas Fonseca
Contact: lucasafonseca@lara.unb.br
Date: Feb 25th 2019
'''

import matplotlib.pyplot as plt
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data_processing import GetFilesToLoad, resample_series, IMU, div_filter, calculate_accel
from PyQt5.QtWidgets import QApplication
from data_classification import *
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy.signal import medfilt
import logging
# import quaternion
import math
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D


normal_plot = True
dash_plot = False

number_of_points = 3
# number_of_points_diff = number_of_points
# filter_size = 49
confidence_level = 0.7

imu_forearm_id = 5
imu_arm_id = 8

imu_0 = 0
# imu_1 = 2
imu_1 = 1

initial_time = 60
total_time = 110


###############################################################################################
###############################################################################################

# Data load

###############################################################################################
###############################################################################################

# sys.stdout = open('Data/results.txt', 'w')

# Choose file
app = QApplication(sys.argv)
source_file = GetFilesToLoad()
app.processEvents()
filename = source_file.filename[0][0]

# filename = 'Data/Estevao_rowing.out'
# filename = 'Data/breno_1604_02.out'

plt.rcParams['svg.fonttype'] = 'none'
logging.basicConfig(filename='Data/results.txt', level=logging.DEBUG)

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
# [emg_1_timestamp, emg_1_values] = [data['emg_1_timestamp'], data['emg_1_values']]
# [emg_2_timestamp, emg_2_values] = [data['emg_2_timestamp'], data['emg_2_values']]

print('Resampling and synchronizing...')
[t, imus[imu_0].resampled_x, imus[imu_1].resampled_x] = resample_series(imus[imu_0].timestamp,
                                                                        imus[imu_0].x_values,
                                                                        imus[imu_1].timestamp,
                                                                        imus[imu_1].x_values)
[t, imus[imu_0].resampled_y, imus[imu_1].resampled_y] = resample_series(imus[imu_0].timestamp,
                                                                        imus[imu_0].y_values,
                                                                        imus[imu_1].timestamp,
                                                                        imus[imu_1].y_values)
[t, imus[imu_0].resampled_z, imus[imu_1].resampled_z] = resample_series(imus[imu_0].timestamp,
                                                                        imus[imu_0].z_values,
                                                                        imus[imu_1].timestamp,
                                                                        imus[imu_1].z_values)
[t, imus[imu_0].resampled_w, imus[imu_1].resampled_w] = resample_series(imus[imu_0].timestamp,
                                                                        imus[imu_0].w_values,
                                                                        imus[imu_1].timestamp,
                                                                        imus[imu_1].w_values)
[t, imus[imu_0].resampled_acc_x, imus[imu_1].resampled_acc_x] = resample_series(imus[imu_0].timestamp,
                                                                                imus[imu_0].acc_x,
                                                                                imus[imu_1].timestamp,
                                                                                imus[imu_1].acc_x)
[t, imus[imu_0].resampled_acc_y, imus[imu_1].resampled_acc_y] = resample_series(imus[imu_0].timestamp,
                                                                                imus[imu_0].acc_y,
                                                                                imus[imu_1].timestamp,
                                                                                imus[imu_1].acc_y)
[t, imus[imu_0].resampled_acc_z, imus[imu_1].resampled_acc_z] = resample_series(imus[imu_0].timestamp,
                                                                                imus[imu_0].acc_z,
                                                                                imus[imu_1].timestamp,
                                                                                imus[imu_1].acc_z)

print('Resampling done')


def make_quaternions(imu):
    q = []
    for i in range(len(imu.resampled_x)):
        q.append(Quaternion(imu.resampled_w[i],
                            imu.resampled_x[i],
                            imu.resampled_y[i],
                            imu.resampled_z[i]
                            ))
    return q


q0 = make_quaternions(imus[imu_0])
q1 = make_quaternions(imus[imu_1])

q = []
[q.append(i * j.conjugate) for i, j in zip(q0, q1)]

qx = []
qy = []
qz = []
qw = []
qang = []
dqang = []
acc_x = [i for i in imus[imu_0].resampled_acc_x]
acc_y = [i for i in imus[imu_0].resampled_acc_y]
acc_z = [i for i in imus[imu_0].resampled_acc_z]

def angle(q):
    try:
        qr = q.elements[0]
        if qr > 1:
            qr = 1
        elif qr < -1:
            qr = -1
        angle = 2 * math.acos(qr)
        angle = angle * 180 / math.pi
        if angle > 180:
            new_angle = 360 - angle
        return angle
    except Exception as e:
        print('Exception "' + str(e) + '" in line ' + str(sys.exc_info()[2].tb_lineno))

for quat in q:
    qw.append(quat.elements[0])
    qx.append(quat.elements[1])
    qy.append(quat.elements[2])
    qz.append(quat.elements[3])
    qang.append(angle(quat))


# for i in range(1, len(qang)):
#     dqang.append( (qang[i] - qang[i-1]) / (t[i] - t[i-1]) )

dqang = np.append([0], np.diff(qang)/np.diff(t))


[t_ang, qang_resampled, buttons_values_resampled] = resample_series(t,
                                                                    qang,
                                                                    buttons_timestamp,
                                                                    buttons_values)

dqang_resampled = []
for i in range(1, len(qang_resampled)):
    dqang_resampled.append( (qang_resampled[i] - qang_resampled[i-1]) / (t_ang[i] - t_ang[i-1]) )


# sys.exit()

###############################################################################################
###############################################################################################

# Aux data calculation

###############################################################################################
###############################################################################################

[qang_low, qang_zero, qang_up] = classify_by_buttons(buttons_timestamp, buttons_values, t, qang)
[dqang_low, dqang_zero, dqang_up] = classify_by_buttons(buttons_timestamp, buttons_values, t[1:], dqang)

qang_avg_low = []
dqang_last_low = []
qang_avg_low_timestamp = []
dqang_last_low_timestamp = []

qang_avg_zero = []
dqang_last_zero= []
qang_avg_zero_timestamp = []
dqang_last_zero_timestamp = []

qang_avg_up = []
dqang_last_up = []
qang_avg_up_timestamp = []
dqang_last_up_timestamp = []


for i in range(len(qang_low)):
    if len(qang_low[i].values) > 0:
        qang_avg_low.append(np.mean(qang_low[i].values))
        qang_avg_low_timestamp.append(qang_low[i].timestamp[-1])
for i in range(len(qang_zero)):
    if len(qang_zero[i].values) > 0:
        qang_avg_zero.append(np.mean(qang_zero[i].values))
        qang_avg_zero_timestamp.append(qang_zero[i].timestamp[-1])
for i in range(len(qang_up)):
    if len(qang_up[i].values) > 0:
        qang_avg_up.append(np.mean(qang_up[i].values))
        qang_avg_up_timestamp.append(qang_up[i].timestamp[-1])
for i in range(len(dqang_low)):
    if len(dqang_low[i].values) > 0:
        dqang_last_low.append(dqang_low[i].values[-1])
        dqang_last_low_timestamp.append(dqang_low[i].timestamp[-1])
for i in range(len(dqang_zero)):
    if len(dqang_zero[i].values) > 0:
        dqang_last_zero.append(dqang_zero[i].values[-1])
        dqang_last_zero_timestamp.append(dqang_zero[i].timestamp[-1])
for i in range(len(dqang_up)):
    if len(dqang_up[i].values) > 0:
        dqang_last_up.append(dqang_up[i].values[-1])
        dqang_last_up_timestamp.append(dqang_up[i].timestamp[-1])





###############################################################################################
###############################################################################################

# Data plot

###############################################################################################
###############################################################################################

# fig1, ax1 = plt.subplots()
# plt.title('Quaternions')
# ax1.plot(t, qx, label='x')
# ax1.plot(t, qy, label='y')
# ax1.plot(t, qz, label='z')
# ax1.plot(t, qw, label='w')
# plt.legend()
# ax2 = ax1.twinx()
# ax2.plot(buttons_timestamp, buttons_values, 'k', label='FES')
# plt.legend()


fig2, (ax3, ax5) = plt.subplots(2, 1)
fig2.canvas.set_window_title('Angle')
ax3.plot(t, qang, label='Ang', color='dodgerblue')
plt.title('Angles')
plt.legend()
ax4 = ax3.twinx()
ax4.plot(buttons_timestamp, buttons_values, 'k', label='FES')
plt.legend()

# fig3, ax5 = plt.subplots()

ax5.plot(t[100:-100], dqang[100:-100], label='Ang', color='dodgerblue')
plt.title('Angle diff')
plt.legend()
ax6 = ax5.twinx()
ax6.plot(buttons_timestamp, buttons_values, 'k', label='FES')
plt.legend()

# fig3 = plt.figure()
# plt.plot(qang_resampled, buttons_values_resampled, '.')

factor = 100
qang_short = div_filter(qang_resampled[1:], factor)
dqang_short = div_filter(dqang_resampled, factor)
buttons_values_short = div_filter(buttons_values_resampled[1:], factor)



fig3d = plt.figure('3D plot')
plt.title('Angle x Diff x FES')
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.scatter(qang_short, dqang_short, buttons_values_short)
ax3d.set_xlabel('Angle')
ax3d.set_ylabel('Diff')
ax3d.set_zlabel('FES')
ax3d.set_ylim3d(-500,500)

# fig4 = plt.figure()
# plt.title('Angle x Diff')
# [plt.plot(i.values[-len(j.values):], j.values, 'b.', label='Flexion') for i, j in zip(qang_low[1:], dqang_low[1:])]
# [plt.plot(i.values[-len(j.values):], j.values, 'k.', label='Stop') for i, j in zip(qang_zero[1:], dqang_zero[1:])]
# [plt.plot(i.values[-len(j.values):], j.values, 'r.', label='Extension') for i, j in zip(qang_up[1:], dqang_up[1:])]
# plt.plot(qang_low, dqang_low, '.b')
# plt.plot()
# plt.ylim(-5000, 5000)

div_factor = 1
plt.figure('Learning data')
plt.title('Low - Zero - Up')
# plt.title('Low / {}'.format(div_factor))
[plt.plot(i.timestamp, i.values, 'b') for i in qang_low[1:round(len(qang_low)/div_factor)]]
# plt.figure()
# plt.title('Zero / {}'.format(div_factor))
[plt.plot(i.timestamp, i.values, 'k') for i in qang_zero[1:round(len(qang_zero)/div_factor)]]
# plt.figure()
# plt.title('Up / {}'.format(div_factor))
[plt.plot(i.timestamp, i.values, 'r') for i in qang_up[1:round(len(qang_up)/div_factor)]]

plt.figure('Angle average')
plt.title('Time x feature')
plt.plot(qang_avg_low_timestamp, qang_avg_low, 'b.', label='low')
plt.plot(qang_avg_zero_timestamp, qang_avg_zero, 'k.', label='zero')
plt.plot(qang_avg_up_timestamp, qang_avg_up, 'r.', label='up')
plt.legend()

plt.figure('Last angle diff')
plt.title('Time x feature')
plt.plot(dqang_last_low_timestamp, dqang_last_low, 'b*', label='low')
plt.plot(dqang_last_zero_timestamp, dqang_last_zero, 'k*', label='zero')
plt.plot(dqang_last_up_timestamp, dqang_last_up, 'r*', label='up')
plt.ylim(-500, 500)
plt.legend()


plt.figure('Slow - Feature crossing')
plt.title('Angle avg x diff')
plt.plot(qang_avg_low[0:round(len(qang_low)/div_factor)], dqang_last_low[0:round(len(qang_low)/div_factor)], 'b.')
plt.plot(qang_avg_zero[0:round(len(qang_zero)/div_factor)], dqang_last_zero[0:round(len(qang_zero)/div_factor)], 'k.')
plt.plot(qang_avg_up[0:round(len(qang_up)/div_factor)], dqang_last_up[0:round(len(qang_up)/div_factor)], 'r.')

# plt.figure('Normal - Feature crossing')
# plt.title('Angle avg x diff')
# plt.plot(qang_avg_low[round(len(qang_low)/div_factor):2*round(len(qang_low)/div_factor)], dqang_last_low[round(len(qang_low)/div_factor):2*round(len(qang_low)/div_factor)], 'b.')
# plt.plot(qang_avg_zero[round(len(qang_low)/div_factor):2*round(len(qang_zero)/div_factor)], dqang_last_zero[round(len(qang_low)/div_factor):2*round(len(qang_zero)/div_factor)], 'k.')
# plt.plot(qang_avg_up[round(len(qang_low)/div_factor):2*round(len(qang_up)/div_factor)], dqang_last_up[round(len(qang_low)/div_factor):2*round(len(qang_up)/div_factor)], 'r.')

# plt.figure('Fast - Feature crossing')
# plt.title('Angle avg x diff')
# plt.plot(qang_avg_low[round(len(qang_low)/div_factor):], dqang_last_low[round(len(qang_low)/div_factor):], 'b.')
# plt.plot(qang_avg_zero[round(len(qang_zero)/div_factor):], dqang_last_zero[round(len(qang_zero)/div_factor):], 'k.')
# plt.plot(qang_avg_up[round(len(qang_up)/div_factor):], dqang_last_up[round(len(qang_up)/div_factor):], 'r.')

# plt.show()
# quit()


###############################################################################################
###############################################################################################

# Machine learning

###############################################################################################
###############################################################################################


# [t, imus[2].resampled_euler_z, imus[0].resampled_euler_z] = resample_series(imus[2].timestamp,
#                                                                             imus[2].euler_z,
#                                                                             imus[0].timestamp,
#                                                                             imus[0].euler_z)
# [t, imus[2].resampled_euler_x, imus[0].resampled_euler_x] = resample_series(imus[2].timestamp,
#                                                                             imus[2].euler_x,
#                                                                             imus[0].timestamp,
#                                                                             imus[0].euler_x)
# [t, imus[2].resampled_euler_y, imus[0].resampled_euler_y] = resample_series(imus[2].timestamp,
#                                                                             imus[2].euler_y,
#                                                                             imus[0].timestamp,
#                                                                             imus[0].euler_y)

# [low, zero, up] = classify_by_buttons(buttons_timestamp, buttons_values, imus[2].timestamp, imus[2].euler_z)

classification0 = classify_by_buttons_in_order(buttons_timestamp, buttons_values, t)



# dqx0 = np.append([0], np.diff(imus[imu_0].resampled_x)/np.diff(t))
# dqx2 = np.append([0], np.diff(imus[imu_1].resampled_x)/np.diff(t))
# dqy0 = np.append([0], np.diff(imus[imu_0].resampled_y)/np.diff(t))
# dqy2 = np.append([0], np.diff(imus[imu_1].resampled_y)/np.diff(t))
# dqz0 = np.append([0], np.diff(imus[imu_0].resampled_z)/np.diff(t))
# dqz2 = np.append([0], np.diff(imus[imu_1].resampled_z)/np.diff(t))
# dqw0 = np.append([0], np.diff(imus[imu_0].resampled_w)/np.diff(t))
# dqw2 = np.append([0], np.diff(imus[imu_1].resampled_w)/np.diff(t))

# dz0 = np.append([0], np.diff(imus[0].resampled_euler_z)/np.diff(t))
# dz2 = np.append([0], np.diff(imus[2].resampled_euler_z)/np.diff(t))
# dx0 = np.append([0], np.diff(imus[0].resampled_euler_x)/np.diff(t))
# dx2 = np.append([0], np.diff(imus[2].resampled_euler_x)/np.diff(t))
# dy0 = np.append([0], np.diff(imus[0].resampled_euler_y)/np.diff(t))
# dy2 = np.append([0], np.diff(imus[2].resampled_euler_y)/np.diff(t))


def find_transitions(timestamp, values):
    transitions_times = []
    transitions_values = []
    for i in range(1, len(timestamp)):
        if values[i] != values[i-1]:
            transitions_times.append(timestamp[i])
            transitions_values.append(values[i])
    return transitions_times, transitions_values

# This method calculates how many predicted transitions were correct within a tolerance to real transitions
def calculate_performance(real_time, real_value, predicted_time, predicted_value, tolerance):
    original_real_time = real_time.copy()
    original_real_value = real_value.copy()
    original_predicted_times = predicted_time.copy()
    original_predicted_values = predicted_value.copy()
    hits = 0
    error_history = []
    total_real_transitions = len(real_time)
    total_predicted_transitions = len(predicted_time)
    false_transitions = 0
    found_transitions_times = []
    found_transitions_values = []
    false_transitions_times = []
    false_transitions_values = []
    wrong_transitions_times = []
    wrong_transitions_values = []
    wrong_transitions = 0
    try:
        transition_counter = 0
        while len(real_time) > 0:
            # current_transition_value = predicted_value[transition_counter]
            # current_transition_time = predicted_time[transition_counter]
            # transition_idx = np.abs(np.asarray(real_time) - current_transition_time).argmin()
            predicted_transition_idx = np.abs(np.asarray(predicted_time) - real_time[0]).argmin()
            current_transition_value = predicted_value[predicted_transition_idx]
            current_transition_time = predicted_time[predicted_transition_idx]
            error = np.abs(real_time[0] - current_transition_time)
            # error_history.append(error)
            if error < tolerance and current_transition_value == real_value[0]:
                hits += 1
                found_transitions_times.append(current_transition_time)
                found_transitions_values.append(current_transition_value)
                del predicted_value[predicted_transition_idx]
                del predicted_time[predicted_transition_idx]
                del real_value[0]
                del real_time[0]
                # del error_history[-1]
                # continue
            # elif transition_counter >= 1 and len(error_history) >= 2:
            #     if current_transition_value == predicted_value[transition_counter - 2] \
            #             and np.abs(current_transition_time - predicted_time[transition_counter - 1]) < tolerance:
            #         false_transitions += 1
            #         false_transitions_times.append(predicted_time[transition_counter-1])
            #         false_transitions_times.append(current_transition_time)
            #         false_transitions_values.append(predicted_value[transition_counter-1])
            #         false_transitions_values.append(current_transition_value)
            #         del error_history[-1]
            #         del error_history[-2]
            #         del predicted_time[-1]
            #         del predicted_time[-2]
            #         del predicted_value[-1]
            #         del predicted_value[-2]
            #         wrong_transitions -= 1
            #         continue
            #     else:
            #         wrong_transitions += 1
            #         wrong_transitions_times.append(current_transition_time)
            #         wrong_transitions_values.append(current_transition_value)
            else:
                wrong_transitions += 1
                wrong_transitions_times.append(current_transition_time)
                wrong_transitions_values.append(current_transition_value)
                del real_time[0]
                del real_value[0]
                error_history.append(error)
            transition_counter += 1
    except Exception as e:
        print('Finished calculating performance after analysing {} transitions'.format(transition_counter))


    # false_transitions = total_predicted_transitions - total_real_transitions
    # false_prediction_time = []
    # false_prediction_value = []
    # for i in range(false_transitions):
    #     idx = np.asarray(error_history).argmax()
    #     false_prediction_time.append(predicted_time[idx])
    #     false_prediction_value.append(predicted_value[idx])
    #     del error_history[idx]

    # if total_real_transitions > total_predicted_transitions:
    #     performance = round(hits/total_real_transitions*100, 2)
    # else:
    #     performance = round(hits / total_predicted_transitions * 100, 2)

    # print('False predicted times: {}'.format(false_prediction_time))
    # print('False predicted values: {}'.format(false_prediction_value))

    performance = round(hits / total_real_transitions * 100, 2)
    print('Total real transitions: {}'.format(total_real_transitions))
    # print(real_time)
    # print(real_value)
    # print(original_real_time)
    # print(original_real_value)
    print('Total predicted transitions: {}'.format(total_predicted_transitions))
    # print(predicted_time)
    # print(predicted_value)
    # print(original_predicted_times)
    # print(original_predicted_values)
    print('Total hits: {}'.format(hits))
    # print(found_transitions_times)
    # print(found_transitions_values)
    # print('Tolerated false Transitions: {}'.format(false_transitions))
    # print(false_transitions_times)
    # print(false_transitions_values)
    print('Wrong transitions: {}'.format(wrong_transitions))
    # print(wrong_transitions_times)
    # print(wrong_transitions_values)
    print('False transitions: {}'.format(total_predicted_transitions - hits - wrong_transitions))
    print('Performance by transitions: {}%'.format(performance))
    print('Mean error: {}s'.format(round(np.mean(error_history), 2)))
    # print(error_history)
    return performance, error_history, false_transitions, predicted_time, predicted_value


def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        for piece_of_data in data:
            pickle.dump(piece_of_data, f)




training_lower_time_table = [initial_time] # [200, 300, 400, 500, 600]
training_upper_time_table = [total_time] # [round(total_time * 3 / 4)] # [275, 375, 475, 575, 675]
testing_lower_time_table = training_lower_time_table # training_upper_time_table
testing_upper_time_table = [total_time] # [300, 400, 500, 600, 700]
tolerance_table = [0.3] # [0.1, 0.2, 0.3, 0.4, 0.5]
# training_lower_time_table = [500, 600]
# training_upper_time_table = [575, 675]
# testing_lower_time_table = training_upper_time_table
# testing_upper_time_table = [600, 700]
# tolerance_table = [0.3, 0.4, 0.5]
total_error = [[], [], [], [], []]

print('\n\n\n')
for trial in range(len(training_lower_time_table)):

    X = []
    y = []
    training_lower_time = training_lower_time_table[trial]
    training_upper_time = training_upper_time_table[trial]
    testing_lower_time = testing_lower_time_table[trial]
    testing_upper_time = testing_upper_time_table[trial]

    for tolerance in tolerance_table:

        print('Starting analysis for time frame {}s-{}s with tolerance of {}s'.format(training_lower_time,
                                                                                      testing_upper_time,
                                                                                      tolerance))
        print('Learning...')
        # TODO: save these values on file
        # TODO: think what to do in the beginning, when there is not enough data
        # Building learning data
        for i in range(number_of_points, len(t)):
            if training_lower_time < t[i] < training_upper_time:
                this = []
                # adding number_of_points points
                # joint angle
                this.append(np.mean(qang[i - number_of_points:i]))

                # quaternions
                # this += [j for j in imus[imu_0].resampled_x[i - number_of_points:i]]
                # this += [j for j in imus[imu_1].resampled_x[i - number_of_points:i]]
                # this += [j for j in imus[imu_0].resampled_y[i - number_of_points:i]]
                # this += [j for j in imus[imu_1].resampled_y[i - number_of_points:i]]
                # this += [j for j in imus[imu_0].resampled_z[i - number_of_points:i]]
                # this += [j for j in imus[imu_1].resampled_z[i - number_of_points:i]]
                # this += [j for j in imus[imu_0].resampled_w[i - number_of_points:i]]
                # this += [j for j in imus[imu_1].resampled_w[i - number_of_points:i]]

                # Euler
                # this += [j for j in imus[0].resampled_euler_z[i - number_of_points:i]]
                # this += [j for j in imus[2].resampled_euler_z[i - number_of_points:i]]
                # this += [j for j in imus[0].resampled_euler_x[i - number_of_points:i]]
                # this += [j for j in imus[2].resampled_euler_x[i - number_of_points:i]]
                # this += [j for j in imus[0].resampled_euler_y[i - number_of_points:i]]
                # this += [j for j in imus[2].resampled_euler_y[i - number_of_points:i]]

                # adding number_of_points diff points
                # joint angle
                this.append(dqang[i])

                this.append(calculate_accel(acc_x, acc_y, acc_z, i))
                # quaternions
                # this += list(dqx0[i - number_of_points:i])
                # this += list(dqx2[i - number_of_points:i])
                # this += list(dqy0[i - number_of_points:i])
                # this += list(dqy2[i - number_of_points:i])
                # this += list(dqz0[i - number_of_points:i])
                # this += list(dqz2[i - number_of_points:i])
                # this += list(dqw0[i - number_of_points:i])
                # this += list(dqw2[i - number_of_points:i])

                # Euler
                # this += [dz0[i]]
                # this += [dz2[i]]
                # this += [dx0[i]]
                # this += [dx2[i]]
                # this += [dy0[i]]
                # this += [dy2[i]]

                X.append(this)

                y.append(classification0[i])

        # Training
        classifier = LinearDiscriminantAnalysis()
        classifier.fit_transform(X, y)


        print('Learning complete')

        # Building evaluating data
        # joint angle
        out = []
        # out_qang = [np.mean(qang[:number_of_points])]
        # out_dqang = [dqang[number_of_points]]

        # quaternions
        # out_qx_0 = [np.array(imus[imu_0].resampled_x[:-number_of_points])]
        # out_qx_2 = [np.array(imus[imu_1].resampled_x[:-number_of_points])]
        # out_qy_0 = [np.array(imus[imu_0].resampled_y[:-number_of_points])]
        # out_qy_2 = [np.array(imus[imu_1].resampled_y[:-number_of_points])]
        # out_qz_0 = [np.array(imus[imu_0].resampled_z[:-number_of_points])]
        # out_qz_2 = [np.array(imus[imu_1].resampled_z[:-number_of_points])]
        # out_qw_0 = [np.array(imus[imu_0].resampled_w[:-number_of_points])]
        # out_qw_2 = [np.array(imus[imu_1].resampled_w[:-number_of_points])]
        #
        # out_dqx0 = [np.array(dqx0[:-number_of_points])]
        # out_dqx2 = [np.array(dqx2[:-number_of_points])]
        # out_dqy0 = [np.array(dqy0[:-number_of_points])]
        # out_dqy2 = [np.array(dqy2[:-number_of_points])]
        # out_dqz0 = [np.array(dqz0[:-number_of_points])]
        # out_dqz2 = [np.array(dqz2[:-number_of_points])]
        # out_dqw0 = [np.array(dqw0[:-number_of_points])]
        # out_dqw2 = [np.array(dqw2[:-number_of_points])]


        # out_z_0 = [np.array(imus[0].resampled_euler_z[:-number_of_points])]
        # out_z_2 = [np.array(imus[2].resampled_euler_z[:-number_of_points])]
        # out_x_0 = [np.array(imus[0].resampled_euler_x[:-number_of_points])]
        # out_x_2 = [np.array(imus[2].resampled_euler_x[:-number_of_points])]
        # out_y_0 = [np.array(imus[0].resampled_euler_y[:-number_of_points])]
        # out_y_2 = [np.array(imus[2].resampled_euler_y[:-number_of_points])]
        if number_of_points > 1:
            for i in range(0, len(qang) - number_of_points):
                # joint angle
                out.append([np.mean(qang[i:number_of_points+i]),
                            dqang[number_of_points+i],
                            calculate_accel(acc_x, acc_y, acc_z, number_of_points+i)
                            ])
                # out_qang = np.append(out_qang, [np.mean(qang[i:number_of_points + i])], 0)
                # out_dqang = np.append(out_dqang, [dqang[number_of_points_diff + i]], 0)

                # quaternions
                # out_qx_0 = np.append(out_qx_0, [np.array(imus[imu_0].resampled_x[i:-number_of_points + i])], 0)
                # out_qx_2 = np.append(out_qx_2, [np.array(imus[imu_1].resampled_x[i:-number_of_points + i])], 0)
                # out_qy_0 = np.append(out_qy_0, [np.array(imus[imu_0].resampled_y[i:-number_of_points + i])], 0)
                # out_qy_2 = np.append(out_qy_2, [np.array(imus[imu_1].resampled_y[i:-number_of_points + i])], 0)
                # out_qz_0 = np.append(out_qz_0, [np.array(imus[imu_0].resampled_z[i:-number_of_points + i])], 0)
                # out_qz_2 = np.append(out_qz_2, [np.array(imus[imu_1].resampled_z[i:-number_of_points + i])], 0)
                # out_qw_0 = np.append(out_qw_0, [np.array(imus[imu_0].resampled_w[i:-number_of_points + i])], 0)
                # out_qw_2 = np.append(out_qw_2, [np.array(imus[imu_1].resampled_w[i:-number_of_points + i])], 0)
                #
                # out_dqx0 = np.append(out_dqx0, [np.array(dqx0[i:-number_of_points + i])], 0)
                # out_dqx2 = np.append(out_dqx2, [np.array(dqx2[i:-number_of_points + i])], 0)
                # out_dqy0 = np.append(out_dqy0, [np.array(dqy0[i:-number_of_points + i])], 0)
                # out_dqy2 = np.append(out_dqy2, [np.array(dqy2[i:-number_of_points + i])], 0)
                # out_dqz0 = np.append(out_dqz0, [np.array(dqz0[i:-number_of_points + i])], 0)
                # out_dqz2 = np.append(out_dqz2, [np.array(dqz2[i:-number_of_points + i])], 0)
                # out_dqw0 = np.append(out_dqw0, [np.array(dqw0[i:-number_of_points + i])], 0)
                # out_dqw2 = np.append(out_dqw2, [np.array(dqw2[i:-number_of_points + i])], 0)

                # out_z_0 = np.append(out_z_0, [np.array(imus[0].resampled_euler_z[i:-number_of_points + i])], 0)
                # out_z_2 = np.append(out_z_2, [np.array(imus[2].resampled_euler_z[i:-number_of_points + i])], 0)
                # out_x_0 = np.append(out_x_0, [np.array(imus[0].resampled_euler_x[i:-number_of_points + i])], 0)
                # out_x_2 = np.append(out_x_2, [np.array(imus[2].resampled_euler_x[i:-number_of_points + i])], 0)
                # out_y_0 = np.append(out_y_0, [np.array(imus[0].resampled_euler_y[i:-number_of_points + i])], 0)
                # out_y_2 = np.append(out_y_2, [np.array(imus[2].resampled_euler_y[i:-number_of_points + i])], 0)

        # joint angle
        # out = np.append(out_qang, out_dqang, 1)

        # quaternions
        # out = np.append(out_qx_0, out_qx_2, 0)
        # out = np.append(out, out_qy_0, 0)
        # out = np.append(out, out_qy_2, 0)
        # out = np.append(out, out_qz_0, 0)
        # out = np.append(out, out_qz_2, 0)
        # out = np.append(out, out_qw_0, 0)
        # out = np.append(out, out_qw_2, 0)
        # out = np.append(out, out_dqx0, 0)
        # out = np.append(out, out_dqx2, 0)
        # out = np.append(out, out_dqy0, 0)
        # out = np.append(out, out_dqy2, 0)
        # out = np.append(out, out_dqz0, 0)
        # out = np.append(out, out_dqz2, 0)
        # out = np.append(out, out_dqw0, 0)
        # out = np.append(out, out_dqw2, 0)

        # out = np.append(out_z_0, out_z_2, 0)
        # out = np.append(out, out_x_0, 0)
        # out = np.append(out, out_x_2, 0)
        # out = np.append(out, out_y_0, 0)
        # out = np.append(out, out_y_2, 0)
        # out = np.append(out, [dz0[number_of_points:]], 0)
        # out = np.append(out, [dz2[number_of_points:]], 0)
        # out = np.append(out, [dx0[number_of_points:]], 0)
        # out = np.append(out, [dx2[number_of_points:]], 0)
        # out = np.append(out, [dy0[number_of_points:]], 0)
        # out = np.append(out, [dy2[number_of_points:]], 0)

        # out = list(out.T)
        save_to_file([X, y, out], 'Data/classifier')


        # Predictions
        print('Calculating predictions')
        predicted_values = classifier.predict(out)
        # predicted_values = medfilt(predicted_values, filter_size)
        print('Predictions calculated')

        scores = classifier.decision_function(X)
        predicted_proba = classifier.predict_proba(out)
        probability = [max(i) for i in predicted_proba]


        trusted_t = []
        trusted_predictions = []
        trusted_classifcation = []
        all_probabilities = [[],[],[]]
        for i in range(len(predicted_values)):
            this_classification = classification0[i]
            if this_classification == -1:
                all_probabilities[0].append(probability[i])
            elif this_classification == 0:
                all_probabilities[1].append(probability[i])
            elif this_classification == 1:
                all_probabilities[2].append(probability[i])
            if probability[i] > confidence_level:
                trusted_t.append(t[i + number_of_points])
                trusted_predictions.append(predicted_values[i])
                trusted_classifcation.append(classification0[i])

        print('Probabilities: ')
        print('-1: {} ({})'.format(np.mean(all_probabilities[0]), np.std(all_probabilities[0])))
        print('0: {} ({})'.format(np.mean(all_probabilities[1]), np.std(all_probabilities[1])))
        print('1: {} ({})'.format(np.mean(all_probabilities[2]), np.std(all_probabilities[2])))


        # plt.figure('Function')
        # plt.plot(decision)
        # plt.figure('Probability')
        # plt.plot(probability)


        # Evaluation
        # print('Evaluating...')
        # evaluated_buttons_timestamp = []
        # evaluated_buttons_values = []
        # evaluated_predicted_time = []
        # evaluated_predicted_values = []
        # for i in range(len(buttons_timestamp)):
        #     if testing_lower_time < buttons_timestamp[i] < testing_upper_time:
        #         evaluated_buttons_timestamp.append(buttons_timestamp[i])
        #         evaluated_buttons_values.append(buttons_values[i])
        # for i in range(len(t) - filter_size - 1):
        #     if testing_lower_time < t[i] < testing_upper_time:
        #         evaluated_predicted_time.append(t[i + filter_size])
        #         evaluated_predicted_values.append(predicted_values[i])
        #
        # [real_transitions_times, real_transitions_values] = find_transitions(evaluated_buttons_timestamp,
        #                                                                      evaluated_buttons_values)
        # [predicted_transitions_times, predicted_transitions_values] = find_transitions(evaluated_predicted_time,
        #                                                                                evaluated_predicted_values)
        #
        # [performance, error, false_transitions, false_predicted_time, false_predicted_values] = calculate_performance(real_transitions_times,
        #                                              real_transitions_values,
        #                                              predicted_transitions_times,
        #                                              predicted_transitions_values,
        #                                              tolerance)
        # total_error[int(tolerance * 10)-1].append(error)
        # plt.figure()
        # plt.hist(error, bins=10)
        # plt.title('Time frame: {}s-{}s. Tolerance: {}s.'.format(training_lower_time, testing_upper_time, tolerance))
        # plt.savefig('{}s-{}s_tolerance_{}.svg'.format(training_lower_time, testing_upper_time, tolerance))
        # plt.show()

        # Here performance point-by-point is calculated, where the classifier is evaluated at every instant,
        # and not only on transitions
        performance = 0
        total = 0
        for i in range(len(trusted_classifcation)):
            if testing_lower_time < trusted_t[i] < testing_upper_time:
                if trusted_predictions[i] == trusted_classifcation[i]:
                    performance += 1
                total += 1
        print('Performance point-by-point: {}%'.format(np.round(performance/total*100, 2)))
        print('##########################################################################################\n\n\n\n')

        ###############################################################################################
        ###############################################################################################

        # Result plotting

        ###############################################################################################
        ###############################################################################################


        # Plots
        if normal_plot:
            print('Plotting...')
            print('IMU 0: {}'.format(imus[imu_0].id))
            print('IMU 1: {}'.format(imus[imu_1].id))

            plt.figure()
            # print('IMU 2: {}'.format(imus[2].id))
            plt.step(buttons_timestamp, buttons_values, 'k', label='FES')
            # plt.plot(imus[1].timestamp, imus[1].euler_x, 'b-')
            # plt.plot(imus[1].timestamp, imus[1].euler_y, 'b:')
            # plt.plot(imus[1].timestamp, imus[1].euler_z, 'b--')
            # plt.plot(imus[0].timestamp, imus[0].euler_x, 'g')
            # plt.plot(imus[0].timestamp, imus[0].euler_y, 'g', label='IMU 0 y')
            # plt.plot(imus[0].timestamp, imus[0].euler_z, 'g', label='IMU 2 z')
            # plt.plot(t, imus[0].resampled_euler_z, 'g', label='IMU 0 z')
            # plt.plot(imus[2].timestamp, imus[2].euler_x, 'b')
            # plt.plot(imus[2].timestamp, imus[2].euler_y, 'b', label= 'IMU 2 y')
            # plt.plot(imus[2].timestamp, imus[2].euler_z, 'b', label='IMU 2 z')
            # plt.plot(t, imus[2].resampled_euler_z, 'b', label='IMU 2 z')
            # [plt.plot(packet.timestamp, packet.values, 'b.', label='Flexion') for packet in low]
            # [plt.plot(packet.timestamp, packet.values, 'g.', label='Stop') for packet in zero]
            # [plt.plot(packet.timestamp, packet.values, 'r.', label='Extension') for packet in up]
            # plt.plot(t, classification0, 'c')
            # plt.plot(t[number_of_points:], predicted_values, 'g:', label='Predicted')
            plt.step(trusted_t, trusted_predictions, 'g-', where='post', label='Trusted predictions')
            # plt.step(false_predicted_time, false_predicted_values, 'b--', label='False predictions')
            # plt.plot(imus[0].timestamp, imus[0].euler_x, 'r-')
            # plt.plot(imus[0].timestamp, imus[0].euler_y, 'r:')
            # plt.plot(imus[0].timestamp, imus[0].euler_z, 'r--')
            # plt.plot(emg_1_timestamp, emg_1_values, 'm-')
            # plt.plot(emg_2_timestamp, emg_2_values, 'm:')
            # plt.plot(imu_2_z_up_timestamp, imu_2_z_up_values, 'r.', label='extension')
            # plt.plot(imu_2_z_zero_timestamp, imu_2_z_zero_values, 'g.', label='stop')
            # plt.plot(imu_2_z_low_timestamp, imu_2_z_low_values, 'b.', label='flexion')

            plt.title(filename)
            plt.legend()
            # legend_elements = [Line2D([0], [0], color='b', label = 'Flexion', marker='o'),
            #                   Line2D([0], [0], color='g', label='Stop', marker='o'),
            #                   Line2D([0], [0], color='r', label='Extension', marker='o')]
            # plt.legend(handles=legend_elements)

            plt.show()

        if dash_plot:

            app_dash = dash.Dash()

            app_dash.layout = html.Div(children=[
                html.Label('Data to graph:'),
                dcc.Checklist(
                    id='data-to-plot',
                    options=[
                        {'label': 'Buttons', 'value': 'buttons'},
                        {'label': 'IMU 0 - x', 'value': 'imus0x'},
                        {'label': 'IMU 0 - y', 'value': 'imus0y'},
                        {'label': 'IMU 0 - z', 'value': 'imus0z'},
                        {'label': 'IMU 1 - x', 'value': 'imus1x'},
                        {'label': 'IMU 1 - y', 'value': 'imus1y'},
                        {'label': 'IMU 1 - z', 'value': 'imus1z'},
                        {'label': 'IMU 2 - x', 'value': 'imus2x'},
                        {'label': 'IMU 2 - y', 'value': 'imus2y'},
                        {'label': 'IMU 2 - z', 'value': 'imus2z'},
                        {'label': 'EMG 1', 'value': 'emg1'},
                        {'label': 'EMG 2', 'value': 'emg2'}
                    ],
                    values=[],
                    style={'display': 'inline-block'}
                ),
                html.Div(id='output-graph'),

            ])


            @app_dash.callback(
                Output(component_id='output-graph', component_property='children'),
                [Input(component_id='data-to-plot', component_property='values')]
            )
            def update_value(input_data):
                #     buttons_to_plot = False
                #     emg_to_plot = [False, False]
                #     imus_to_plot = [False, False, False]

                graph_data = []

                if 'buttons' in input_data:
                    # buttons = True
                    include = [{'x': buttons_timestamp, 'y': buttons_values, 'name': 'buttons'}]
                    graph_data = graph_data + include
                if 'imus0x' in input_data:
                    # imus[0] = True
                    include = [{'x': imus[0].timestamp, 'y': imus[0].x_values, 'name': 'imu0x'}]
                    graph_data = graph_data + include
                if 'imus0y' in input_data:
                    include = [{'x': imus[0].timestamp, 'y': imus[0].y_values, 'name': 'imu0y'}]
                    graph_data = graph_data + include
                if 'imus0z' in input_data:
                    include = [{'x': imus[0].timestamp, 'y': imus[0].z_values, 'name': 'imu0z'}]
                    graph_data = graph_data + include
                if 'imus1x' in input_data:
                    # imus[1] = True
                    include = [{'x': imus[1].timestamp, 'y': imus[1].x_values, 'name': 'imus1x'}]
                    graph_data = graph_data + include
                if 'imus1y' in input_data:
                    include = [{'x': imus[1].timestamp, 'y': imus[1].y_values, 'name': 'imus1y'}]
                    graph_data = graph_data + include
                if 'imus1z' in input_data:
                    include = [{'x': imus[1].timestamp, 'y': imus[1].z_values, 'name': 'imus1z'}]
                    graph_data = graph_data + include
                if 'imus2x' in input_data:
                    # imus[2] = True
                    include = [{'x': imus[2].timestamp, 'y': imus[2].x_values, 'name': 'imus2x'}]
                    graph_data = graph_data + include
                if 'imus2y' in input_data:
                    include = [{'x': imus[2].timestamp, 'y': imus[2].y_values, 'name': 'imus2y'}]
                    graph_data = graph_data + include
                if 'imus2z' in input_data:
                    include = [{'x': imus[2].timestamp, 'y': imus[2].z_values, 'name': 'imus2z'},
                               # {'x': imu_2_z_low_timestamp, 'y': imu_2_z_low_values, 'mode': 'markers', 'name': 'IMU 2 flexion'},
                               # {'x': imu_2_z_zero_timestamp, 'y': imu_2_z_zero_values, 'mode': 'markers', 'name': 'IMU 2 stop'},
                               # {'x': imu_2_z_up_timestamp, 'y': imu_2_z_up_values, 'mode': 'markers', 'name': 'IMU 2 extension'}
                               ]
                    graph_data = graph_data + include
                # if 'emg1' in input_data:
                #     emg[0] = True
                    # include = [{'x': emg_1_timestamp, 'y': emg_1_values, 'name': 'emg1'}]
                    # graph_data = graph_data + include
                # if 'emg2' in input_data:
                #     emg[1] = True
                    # include = [{'x': emg_2_timestamp, 'y': emg_2_values, 'name': 'emg2'}]
                    # graph_data = graph_data + include

                return dcc.Graph(
                    id='graph',
                    figure={
                        'data': graph_data,
                        'layout': {
                            'title': 'Rowing data'
                        }
                    },
                    style={'height': 800},
                )


            app_dash.run_server(debug=False)
            # dash_process = multiprocessing.Process(target=run_dash, args=(app_dash,))
            # dash_process.start()

# for i in range(5):
#     plt.figure()
#     plt.hist(total_error[i], bins=10)
#     plt.title('Total error for tolerance = {}s'.format((i + 1) / 10))
#     plt.savefig('total_error_hist_tolerance_{}s.svg'.format((i + 1) / 10))

# plt.show()
