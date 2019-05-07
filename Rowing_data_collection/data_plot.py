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
from data_processing import GetFilesToLoad, resample_series, IMU, div_filter, calculate_accel, correct_fes_input, \
    find_classes_and_transitions, lpf, median_filter
from PyQt5.QtWidgets import QApplication
from data_classification import *
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np
from scipy.signal import medfilt
import logging
import math
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

# mode = 'singleLDA'
mode = 'switchingLDA'
# mode = 'manual'

normal_plot = True
dash_plot = False

# number_of_points = 149
window_size = 1
if mode == 'singleLDA':
    confidence_level = [0.85]
else:
    confidence_level = [0.5, 0.5, 0.5]

# accel filter
filter_acc = True
cutoff = 0.5
fs = 50
filter_size = 3
output_command_filter_size = 1
number_of_stds = 3


imu_forearm_id = 4
imu_arm_id = 5


initial_time = 110
total_time = 160

accel_threshold = 0.05

# classes = [-1, 1, 0]


###############################################################################################
###############################################################################################

# Data load

###############################################################################################
###############################################################################################

# sys.stdout = open('Data/results.txt', 'w')

# Choose file
# app = QApplication(sys.argv)
# source_file = GetFilesToLoad()
# app.processEvents()
# filename = source_file.filename[0][0]

# filename = 'Data/Estevao_rowing.out'
# filename = 'Data/breno_1604_02.out'
# filename = 'Data/lucas_with_accel_01.out'
filename = 'Data/roberto_03.out'

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
if imus[0].id == imu_forearm_id:
    imu_0 = 0
    imu_1 = 1
else:
    imu_1 = 0
    imu_0 = 1

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
avg_f = len(t) / (t[-1] - t[0])
print('Average frequency: {}'.format(avg_f))
number_of_points = round(avg_f * window_size)


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
acc_x_0 = [i for i in imus[imu_0].resampled_acc_x]
acc_y_0 = [i for i in imus[imu_0].resampled_acc_y]
acc_z_0 = [i for i in imus[imu_0].resampled_acc_z]
acc_x_1 = [i for i in imus[imu_1].resampled_acc_x]
acc_y_1 = [i for i in imus[imu_1].resampled_acc_y]
acc_z_1 = [i for i in imus[imu_1].resampled_acc_z]

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

buttons_values = correct_fes_input(buttons_timestamp, buttons_values)

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


classes, trasitions = find_classes_and_transitions(buttons_values, buttons_timestamp, initial_time, total_time)

print('Classes: {}'.format(classes))
print('Transitions: {}'.format(trasitions))

# acc_x_0_filtered = lpf(np.array(acc_x_0), cutoff, fs)
# acc_y_0_filtered = lpf(np.array(acc_y_0), cutoff, fs)
# acc_z_0_filtered = lpf(np.array(acc_z_0), cutoff, fs)
# acc_x_1_filtered = lpf(np.array(acc_x_1), cutoff, fs)
# acc_y_1_filtered = lpf(np.array(acc_y_1), cutoff, fs)
# acc_z_1_filtered = lpf(np.array(acc_z_1), cutoff, fs)
acc_x_0_filtered = medfilt(acc_x_0, filter_size)
acc_y_0_filtered = medfilt(acc_y_0, filter_size)
acc_z_0_filtered = medfilt(acc_z_0, filter_size)
acc_x_1_filtered = medfilt(acc_x_1, filter_size)
acc_y_1_filtered = medfilt(acc_y_1, filter_size)
acc_z_1_filtered = medfilt(acc_z_1, filter_size)


# sys.exit()

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


fig2, (ax3, ax5) = plt.subplots(2, 1, sharex=True)
fig2.canvas.set_window_title('Angle')

ax3.plot(t, qang, label='Ang', color='dodgerblue')
ax3.set_title('Angles')
ax3.legend()
ax3.set_ylabel('degrees')

ax4 = ax3.twinx()
ax4.plot(buttons_timestamp, buttons_values, 'k', label='FES')
ax4.set_yticks([-1, 0, 1])
ax4.legend()
ax4.set_ylabel('Flex=-1, Off=0, Ext=1')

# ax5.plot(t, np.array(acc_x_0_filtered) + 1, 'b', label='x')
# ax5.plot(t, acc_y_0_filtered, 'b', label='y')
ax5.plot(t, np.array(acc_z_0_filtered), 'b', label='z')
# ax5.plot(t, np.array(acc_x_1_filtered), 'g', label='x')
# ax5.plot(t, acc_y_1_filtered, 'g', label='y')
# ax5.plot(t, np.array(acc_z_1_filtered) - 1, 'g', label='z')
ax5.set_title('Accel')
ax5.legend()
ax5.set_ylabel('g')

ax6 = ax5.twinx()
ax6.plot(buttons_timestamp, buttons_values, 'k', label='FES')
ax6.set_yticks([-1, 0, 1])
ax6.legend()
ax6.set_ylabel('Flex=-1, Off=0, Ext=1')

# fig3 = plt.figure()
# plt.plot(qang_resampled, buttons_values_resampled, '.')

factor = 100
qang_short = div_filter(qang_resampled[1:], factor)
dqang_short = div_filter(dqang_resampled, factor)
buttons_values_short = div_filter(buttons_values_resampled[1:], factor)



# fig3d = plt.figure('3D plot')
# plt.title('Angle x Diff x FES')
# ax3d = fig3d.add_subplot(111, projection='3d')
# ax3d.scatter(qang_short, dqang_short, buttons_values_short)
# ax3d.set_xlabel('Angle')
# ax3d.set_ylabel('Diff')
# ax3d.set_zlabel('FES')
# ax3d.set_ylim3d(-500,500)

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

# plt.figure('Angle average')
# plt.title('Time x feature')
# plt.plot(qang_avg_low_timestamp, qang_avg_low, 'b.', label='low')
# plt.plot(qang_avg_zero_timestamp, qang_avg_zero, 'k.', label='zero')
# plt.plot(qang_avg_up_timestamp, qang_avg_up, 'r.', label='up')
# plt.legend()

# plt.figure('Last angle diff')
# plt.title('Time x feature')
# plt.plot(dqang_last_low_timestamp, dqang_last_low, 'b*', label='low')
# plt.plot(dqang_last_zero_timestamp, dqang_last_zero, 'k*', label='zero')
# plt.plot(dqang_last_up_timestamp, dqang_last_up, 'r*', label='up')
# plt.ylim(-500, 500)
# plt.legend()


plt.figure('Feature crossing')
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


def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        for piece_of_data in data:
            pickle.dump(piece_of_data, f)


training_lower_time_table = [initial_time] # [200, 300, 400, 500, 600]
training_upper_time_table = [total_time] # [round(total_time * 3 / 4)] # [275, 375, 475, 575, 675]
testing_lower_time_table = training_lower_time_table # training_upper_time_table
testing_upper_time_table = [total_time] # [300, 400, 500, 600, 700]


trial = 0
training_lower_time = training_lower_time_table[trial]
training_upper_time = training_upper_time_table[trial]
testing_lower_time = testing_lower_time_table[trial]
testing_upper_time = testing_upper_time_table[trial]
total_length = len(classification0)

# training
print('Training')
lda = []
decision_functions = []
scores = []
xs = []
ys = []
new_xs = []

X = []
y = []

if mode == 'singleLDA':
    for j in range(total_length - 1):
        if training_lower_time < t[j] < training_upper_time and j + number_of_points < total_length:
            this = []
            this.append(np.mean(qang[j:j + number_of_points]))
            this.append(np.mean(dqang[j:j + number_of_points]))
            this.append(np.mean(acc_x_0_filtered[j:j + number_of_points]))
            this.append(np.mean(acc_y_0_filtered[j:j + number_of_points]))
            this.append(np.mean(acc_z_0_filtered[j:j + number_of_points]))
            this.append(np.mean(acc_x_1_filtered[j:j + number_of_points]))
            this.append(np.mean(acc_y_1_filtered[j:j + number_of_points]))
            this.append(np.mean(acc_z_1_filtered[j:j + number_of_points]))
            # this.append(acc_x_0[j + number_of_points])
            # this.append(acc_y_0[j + number_of_points])
            # this.append(acc_z_0[j + number_of_points])
            # this.append(acc_x_1[j + number_of_points])
            # this.append(acc_y_1[j + number_of_points])
            # this.append(acc_z_1[j + number_of_points])

            X.append(this)
            y.append(classification0[j + number_of_points])
    # new_lda = LinearDiscriminantAnalysis(store_covariance=True, priors=[0.6, 0.4])
    new_lda = LinearDiscriminantAnalysis(store_covariance=True)
    # # new_lda = QuadraticDiscriminantAnalysis(store_covariance=True, priors=None)
    new_x = new_lda.fit_transform(X, y)
    # # new_lda.fit(X, y)
    xs.append(X)
    ys.append(y)
    new_xs.append(new_x)
    # decision_functions.append(new_lda.decision_function(X))
    scores.append(new_lda.score(X, y))
    lda.append(new_lda)
    #
    # new_x_0 = new_x[np.array(y) == min(y)]
    # new_y_0 = np.array(y)[np.array(y) == min(y)]
    # new_x_1 = new_x[np.array(y) == max(y)]
    # new_y_1 = np.array(y)[np.array(y) == max(y)]

    x = []
    x_means = []
    x_stds = []
    labels = []
    for c in classes:
        x.append(new_x[np.array(y) == c, :])
        x_means.append([np.mean(x[-1][:, 0]), np.mean(x[-1][:, 1])])
        x_stds.append([np.std(x[-1][:, 0]), np.std(x[-1][:, 1])])
        labels.append(str(c))

    fig = plt.figure('Class separation')
    ax = fig.gca()
    for i in range(len(x)):
        plt.scatter(x[i][:,0], x[i][:,1], label=labels[i])
        plt.plot(x_means[i][0], x_means[i][1],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')
        ell = Ellipse(x_means[i], x_stds[i][0] * number_of_stds, x_stds[i][1] * number_of_stds,
                      facecolor='C{}'.format(i), edgecolor='black', linewidth=2)
        # ell.set_clip_box(fig.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    plt.legend()

    # plt.plot(new_x_0_mean, new_y_0[0],
    #          '*', color='yellow', markersize=15, markeredgecolor='grey')
    # plt.plot(new_x_1_mean, new_y_1[0],
    #          '*', color='yellow', markersize=15, markeredgecolor='grey')
    # plt.plot([new_x_0_mean-new_x_0_std, new_x_0_mean+new_x_0_std], [new_y_0[0], new_y_0[0]],
    #          '|', color='red', markersize=30)
    # plt.plot([new_x_1_mean - new_x_1_std, new_x_1_mean + new_x_1_std], [new_y_1[0], new_y_1[0]],
    #          '|', color='red', markersize=30)

else:
    for i in range(len(classes)):
        X = []
        y = []
        if i == len(classes) - 1:
            for j in range(total_length-1):
                if training_lower_time < t[j] < training_upper_time and j + number_of_points < total_length:
                    this = []

                    if classification0[j + number_of_points] == classes[i] or classification0[j + number_of_points] == \
                            classes[0]:
                        this.append(np.mean(qang[j:j + number_of_points]))
                        this.append(np.mean(dqang[j:j + number_of_points]))
                        this.append(np.mean(acc_x_0_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_y_0_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_z_0_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_x_1_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_y_1_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_z_1_filtered[j:j + number_of_points]))

                        X.append(this)
                        y.append(classification0[j + number_of_points])

            # lda[i].fit(X, y)
        else:
            # X = []
            # y = []
            for j in range(total_length-1):
                if training_lower_time < t[j] < training_upper_time and j + number_of_points < total_length:
                    this = []

                    if classification0[j + number_of_points] == classes[i] or classification0[j + number_of_points] == \
                            classes[i + 1]:
                        this.append(np.mean(qang[j:j + number_of_points]))
                        this.append(np.mean(dqang[j:j + number_of_points]))
                        this.append(np.mean(acc_x_0_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_y_0_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_z_0_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_x_1_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_y_1_filtered[j:j + number_of_points]))
                        this.append(np.mean(acc_z_1_filtered[j:j + number_of_points]))

                        X.append(this)
                        y.append(classification0[j+number_of_points])

        # new_lda = LinearDiscriminantAnalysis(store_covariance=True, priors=[0.6, 0.4])
        new_lda = LinearDiscriminantAnalysis(store_covariance=True)
        # # new_lda = QuadraticDiscriminantAnalysis(store_covariance=True, priors=None)
        new_x = new_lda.fit_transform(X, y)
        # # new_lda.fit(X, y)
        xs.append(X)
        ys.append(y)
        new_xs.append(new_x)
        # decision_functions.append(new_lda.decision_function(X))
        scores.append(new_lda.score(X, y))
        lda.append(new_lda)
        #
        new_x_0 = new_x[np.array(y) == min(y)]
        new_y_0 = np.array(y)[np.array(y) == min(y)]
        new_x_1 = new_x[np.array(y) == max(y)]
        new_y_1 = np.array(y)[np.array(y) == max(y)]
        #
        fig = plt.figure('Class Separation')
        splot = plt.subplot(3, 1, i+1)
        plt.title('LDA {}'.format(i))
        #
        # plt.scatter(new_x, y)
        plt.scatter(new_x_0, new_y_0)
        plt.scatter(new_x_1, new_y_1)
        new_x_0_mean = np.mean(new_x_0)
        new_x_1_mean = np.mean(new_x_1)
        new_x_0_std = np.std(new_x_0)
        new_x_1_std = np.std(new_x_1)
        plt.plot(new_x_0_mean, new_y_0[0],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')
        plt.plot(new_x_1_mean, new_y_1[0],
                 '*', color='yellow', markersize=15, markeredgecolor='grey')
        plt.plot([new_x_0_mean - number_of_stds * new_x_0_std, new_x_0_mean + number_of_stds * new_x_0_std],
                 [new_y_0[0], new_y_0[0]],
                 '|', color='red', markersize=30)
        plt.plot([new_x_1_mean - number_of_stds * new_x_1_std, new_x_1_mean + number_of_stds * new_x_1_std],
                 [new_y_1[0], new_y_1[0]],
                 '|', color='red', markersize=30)
        # plt.scatter(np.array(X)[:, 0], np.array(y) + 3)
        plt.ylim([-2, 2])

print('scores: {}'.format(scores))
# plt.show()
# print('Training completed')
# exit()

# confidence_level = np.array([1.5, 1.5, 1.5]) - scores
print('Confidence levels: {}'.format(confidence_level))
# confidence_level = [0.5, 0.5, 0.5]

print('Saving classifier to file...')
# saving trained LDAs and evaluating data
save_to_file([lda, classes, number_of_points, confidence_level], 'Data/classifier2.lda')


###############################################################################################
###############################################################################################

# Simulation

###############################################################################################
###############################################################################################


print('Generating evaluation data...')
# confidence_level = scores
# sys.exit()
# computing evaluating data

if filter_size > number_of_points:
    filter_size = number_of_points
out = []
if number_of_points > 1:
    for i in range(0, len(qang) - number_of_points):
        out.append([
            np.mean(qang[i:number_of_points + i]),
            np.mean(dqang[i:number_of_points + i]),
            np.mean(medfilt(acc_x_0[i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_y_0[i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_z_0[i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_x_1[i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_y_1[i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_z_1[i:number_of_points + i], filter_size))
            # lpf(acc_x_0[i:number_of_points + i], cutoff, fs)[-1],
            # lpf(acc_y_0[i:number_of_points + i], cutoff, fs)[-1],
            # lpf(acc_z_0[i:number_of_points + i], cutoff, fs)[-1],
            # lpf(acc_x_1[i:number_of_points + i], cutoff, fs)[-1],
            # lpf(acc_y_1[i:number_of_points + i], cutoff, fs)[-1],
            # lpf(acc_z_1[i:number_of_points + i], cutoff, fs)[-1]
            # acc_x_0[number_of_points + i],
            # acc_y_0[number_of_points + i],
            # acc_z_0[number_of_points + i],
            # acc_x_1[number_of_points + i],
            # acc_y_1[number_of_points + i],
            # acc_z_1[number_of_points + i]
            ])



    # def probability(self, values):
    #     return max(max(self.lda.predict_proba(np.array(values).reshape(1, -1))))

# c1 = Classifier(lda1)
# c2 = Classifier(lda2)
# c3 = Classifier(lda3)

c = Classifier(lda)

# Predictions
print('Calculating predictions...')
predictions = []
probabilities = []

state = -1
state_prediction = [0 for i in range(output_command_filter_size)]
state_probability = [0 for i in range(output_command_filter_size)]

output_command = []

for value in out:
    [new_prediction, new_probability] = c.classify(value)
    predictions.append(new_prediction)
    probabilities.append(new_probability)
    # print(new_prediction, new_probability)

    if mode == 'manual':
        if state == -1 and (value[4] > accel_threshold): # and value[5] > 0.25):
            state = 1
            state_prediction.append(state)
            state_probability.append(1)
        elif state == 1 and value[0] > 90 and value[1] < 0:
            state = 0
            state_prediction.append(state)
            state_probability.append(1)
        elif state == 0 and value[0] < 15:
            state = -1
            state_prediction.append(state)
            state_probability.append(1)
        else:
            state_prediction.append(state_prediction[-1])
            state_probability.append(state_probability[-1])

    elif mode == 'singleLDA':
        if new_probability[0] > confidence_level[0]:
            state_prediction.append(new_prediction[0])
            state_probability.append(new_probability[0])
        else:
            state_prediction.append(state_prediction[-1])
            state_probability.append(state_probability[-1])

    elif mode == 'switchingLDA':
        for s in classes:
            if state == s:
                i = classes.index(s)
                if new_probability[i] > confidence_level[i]:
                    state = new_prediction[i]
                    state_prediction.append(new_prediction[i])
                    state_probability.append(new_probability[i])
                else:
                    state_prediction.append(state_prediction[-1])
                    state_probability.append(state_probability[-1])
                break
    output_command.append(np.median(state_prediction[-output_command_filter_size:]))
[state_prediction.pop(0) for i in range(output_command_filter_size)]
[state_probability.pop(0) for i in range(output_command_filter_size)]

print('Predictions calculated')

# scores = classifier.decision_function(X)
# predicted_proba = classifier.predict_proba(out)
# probability = [max(i) for i in predicted_proba]




# print('Probabilities: ')
# print('-1: {} ({})'.format(np.mean(all_probabilities[0]), np.std(all_probabilities[0])))
# print('0: {} ({})'.format(np.mean(all_probabilities[1]), np.std(all_probabilities[1])))
# print('1: {} ({})'.format(np.mean(all_probabilities[2]), np.std(all_probabilities[2])))


###############################################################################################
###############################################################################################

# Evaluation

###############################################################################################
###############################################################################################


temp_t = []
temp_prediction = []
temp_truth = []
temp_score = []
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
for i in range(len(output_command)):
    if testing_lower_time < t[i+number_of_points] < testing_upper_time:
        if output_command[i] == classification0[i+number_of_points]:
            performance += 1
            temp_score.append(1)
        else:
            temp_score.append(0)
        total += 1
        temp_t.append(t[i+number_of_points])
        temp_truth.append(classification0[i+number_of_points])
        temp_prediction.append(output_command[i])
print('Point-by-point performance: {}%'.format(np.round(performance/total*100, 2)))
print('##########################################################################################')

###############################################################################################
###############################################################################################

# Result plotting

###############################################################################################
###############################################################################################


# Plots
if normal_plot:
    print('Plotting...')
    # print('IMU 0: {}'.format(imus[imu_0].id))
    # print('IMU 1: {}'.format(imus[imu_1].id))

    if mode == 'switchingLDA':
        # plt.figure()
        fig, ax = plt.subplots(len(classes), 1, sharex=True, sharey=True)
        fig.canvas.set_window_title('Each LDA performance')
        # print('IMU 2: {}'.format(imus[2].id))
        plt.step(buttons_timestamp, buttons_values, 'k', label='FES')
        for i in range(len(classes)):
            ax[i].step(buttons_timestamp, buttons_values, 'k', label='FES')
            ax[i].step(t[number_of_points:], [prediction[i] for prediction in predictions])
            plt.title('LDA {}'.format(i))
        # ax1.step(buttons_timestamp, buttons_values)
        # ax1.step(t[number_of_points:], predictions1)
        # ax2.plot(t[number_of_points:], proba1)
        # plt.title('LDA 0')

        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        # ax1.step(buttons_timestamp, buttons_values)
        # ax1.step(t[number_of_points:], predictions2)
        # ax2.plot(t[number_of_points:], proba2)
        # plt.title('LDA 1')

        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        # ax1.step(buttons_timestamp, buttons_values)
        # ax1.step(t[number_of_points:], predictions3)
        # ax2.plot(t[number_of_points:], proba3)
        # plt.title('LDA 2')

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    # plt.title('Switching states')
    # ax1.step(t, classification0)
    # ax1.step(t[number_of_points:], state_prediction)
    # ax2.plot(t[number_of_points:], state_probability)
    #
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.canvas.set_window_title('Simulation result')
    ax1.step(temp_t, temp_truth, label='truth')
    ax1.step(temp_t, temp_prediction, label='prediction')
    plt.legend()
    plt.title(mode)
    ax2.step(temp_t, temp_score)
    plt.title('Score')

    # [plt.plot(packet.timestamp, packet.values, 'b.', label='Flexion') for packet in low]
    # [plt.plot(packet.timestamp, packet.values, 'g.', label='Stop') for packet in zero]
    # [plt.plot(packet.timestamp, packet.values, 'r.', label='Extension') for packet in up]
    # plt.plot(t, classification0, 'c')
    # plt.plot(t[number_of_points:], predicted_values, 'g:', label='Predicted')
    # plt.step(trusted_t, trusted_predictions, 'g-', where='post', label='Trusted predictions')
    # plt.step(false_predicted_time, false_predicted_values, 'b--', label='False predictions')
    # plt.plot(imu_2_z_up_timestamp, imu_2_z_up_values, 'r.', label='extension')
    # plt.plot(imu_2_z_zero_timestamp, imu_2_z_zero_values, 'g.', label='stop')
    # plt.plot(imu_2_z_low_timestamp, imu_2_z_low_values, 'b.', label='flexion')

    # plt.title(filename)
    # plt.legend()
    # legend_elements = [Line2D([0], [0], color='b', label = 'Flexion', marker='o'),
    #                   Line2D([0], [0], color='g', label='Stop', marker='o'),
    #                   Line2D([0], [0], color='r', label='Extension', marker='o')]
    # plt.legend(handles=legend_elements)
    # plt.switch_backend('Qt5Agg')
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    fig = plt.figure('Frequency analysis')
    plt.plot(t[1:], 1/np.diff(t))
    plt.plot(t[1:], medfilt(1/np.diff(t), 25))
    # print('Average frequency: {}'.format(len(t)/(t[-1]-t[1])))

    plt.show()



# for i in range(5):
#     plt.figure()
#     plt.hist(total_error[i], bins=10)
#     plt.title('Total error for tolerance = {}s'.format((i + 1) / 10))
#     plt.savefig('total_error_hist_tolerance_{}s.svg'.format((i + 1) / 10))

# plt.show()
