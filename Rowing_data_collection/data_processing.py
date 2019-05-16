'''
This file is a collection of useful functions for dealing with rowing data, including open and saving files, IMU data,
EMG data and data syncing.
Author: Lucas Fonseca
Contact: lucasafonseca@lara.unb.br
Date: Feb 25th 2019
'''

from PyQt5.QtWidgets import QWidget, QFileDialog
from transformations import euler_from_quaternion
from pyquaternion import Quaternion
from numpy import mean
import numpy as np
import math
import sys


class GetFileToSave(QWidget):

    def __init__(self):
        super(GetFileToSave, self).__init__()
        self.filename = []
        self.openFileDialog()

    def openFileDialog(self):
        filename = QFileDialog.getSaveFileName(self)
        if filename:
            self.filename = filename


class GetFilesToLoad(QWidget):

    def __init__(self):
        super(GetFilesToLoad, self).__init__()
        self.filename = []
        self.openFileDialog()

    def openFileDialog(self):
        filename = QFileDialog.getOpenFileNames(self)
        if filename:
            self.filename = filename

class IMU:

    def __init__(self, this_id):
        self.id = this_id
        self.timestamp = []
        self.x_values = []
        self.y_values = []
        self.z_values = []
        self.w_values = []
        self.euler_x = []
        self.euler_y = []
        self.euler_z = []
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []

    def get_euler_angles(self):
        for i in range(len(self.timestamp)):
            # [self.euler_x[i], self.euler_y[i], self.euler_z[i]] =
            euler = euler_from_quaternion((self.x_values[i],
                                           self.y_values[i],
                                           self.z_values[i],
                                           self.w_values[i]))
            self.euler_x.append(euler[0])
            self.euler_y.append(euler[1])
            self.euler_z.append(euler[2])


def lpf(x, cutoff, fs, order=5):
    import numpy as np
    from scipy.signal import filtfilt, butter
    """
    low pass filters signal with Butterworth digital
    filter according to cutoff frequency

    filter uses Gustafsson’s method to make sure
    forward-backward filt == backward-forward filt

    Note that edge effects are expected

    Args:
        x      (array): signal data (numpy array)
        cutoff (float): cutoff frequency (Hz)
        fs       (int): sample rate (Hz)
        order    (int): order of filter (default 5)

    Returns:
        filtered (array): low pass filtered data
    """
    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist)
    if not np.all(np.abs(np.roots(a)) < 1):
        raise Exception('Filter with cutoff at {} Hz is unstable given '
                         'sample frequency {} Hz'.format(cutoff, fs))
    filtered = filtfilt(b, a, x, method='gust')
    return filtered

def median_filter(x, size):
    import numpy as np
    if size % 2 != 0:
        raise Exception('size must be odd')
    halfish = (size - 1) / 2
    out = x[0:halfish]
    for i in range(halfish + 1, len(x) - halfish):
        out.append(np.mean(np.array(x[i - halfish:i + halfish])))
    out = out + x[-halfish:]
    if not len(x) == len(out):
        raise Exception('Out vector with different size than input vector')


# Find files with EMG, IMU and button data
def separate_files(filenames):
    emg_files = [f for f in filenames if 'EMG' in f]
    imus_files = [f for f in filenames if 'imu' in f]
    buttons_files = [f for f in filenames if 'stim' in f]
    return [emg_files, imus_files, buttons_files]

def parse_button_file(filename, starting_time):
    lines = []
    timestamp = []
    button_state = []
    with open(filename) as inputfile:
        for line in inputfile:
            lines.append(line.split(','))
    # [print(line) for line in lines]
    # first_time = float(lines[0][0])
    timestamp.append(float(lines[0][0]) - starting_time)
    button_state.append(get_button_value(lines[0][2]))
    for i in range(len(lines[1:])):
        timestamp.append(timestamp[-1])
        button_state.append(get_button_value(lines[i][2]))
        timestamp.append(float(lines[i][0]) - starting_time)
        button_state.append(get_button_value(lines[i][2]))
    # for data in lines:
    #     timestamp.append(float(data[0]) - starting_time)
    #     button_state.append(get_button_value(data[2]))

    return [timestamp, button_state]

def parse_emg_file(filename, starting_time):
    lines = []
    timestamp = []
    emg_data = []
    with open(filename) as inputfile:
        for line in inputfile:
            lines.append(line.split(','))
    # first_time = float(lines[0][0])
    # last_time = 0
    for data in lines[1:]:
        this_time = float(data[0]) - starting_time

        # timestamp = timestamp + list(linspace(last_time, this_time, len(data)))[0:-1]
        timestamp.append(this_time)
        # last_time = this_time
        # [emg_data.append(float(d)) for d in data[1:]]
        this_emg = []
        [this_emg.append(float(i)) for i in data[1:]]
        emg_data.append(filter_emg(this_emg))

    return [timestamp, emg_data]

def parse_imus_file(filename, starting_time=0):
    lines = []
    imus = []
    imus_ids = []
    with open(filename) as inputfile:
        for line in inputfile:
            lines.append(line.split(','))
    # first_time = float(lines[0][0])
    if starting_time == 0:
        starting_time = float(lines[0][0])
    for data in lines[1:]:
        id = float(data[2])
        if id not in imus_ids:
            imus_ids.append(id)
            imus.append(IMU(id))
        imus[imus_ids.index(id)].timestamp.append(float(data[0]) - starting_time)
        imus[imus_ids.index(id)].x_values.append(float(data[3]))
        imus[imus_ids.index(id)].y_values.append(float(data[4]))
        imus[imus_ids.index(id)].z_values.append(float(data[5]))
        imus[imus_ids.index(id)].w_values.append(float(data[6]))
        imus[imus_ids.index(id)].acc_x.append(float(data[7]))
        imus[imus_ids.index(id)].acc_y.append(float(data[8]))
        imus[imus_ids.index(id)].acc_z.append(float(data[9]))

    [imus[i].get_euler_angles() for i in range(len(imus))]

    return imus

def filter_emg(emg_data):
    values_to_pop = []
    j = len(emg_data)
    try:
        for i in range(j):
            if emg_data[j] == -1:
                # values_to_pop.append(i)
                emg_data.pop(i)
            else:

                j = + 1
    except Exception:
        pass
    # TODO implement filter here
    norm = [i/max(emg_data) for i in emg_data]
    return mean(norm)

def get_button_value(button_state):
    if button_state.find('stop') != -1:
        return 0
    elif button_state.find('extension') != -1:
        return 1
    elif button_state.find('flexion') != -1:
        return -1

def get_starting_time(filenames):
    times = []
    for filename in filenames:
        with open(filename) as inputfile:
            for line in inputfile:
                line = line.split(',')
                times.append(float(line[0]))
                break

    return min(times)

def run_dash(app_dash):
    app_dash.run_server(debug=True)

# Method for syncing data from sources with different sample rates, or inconsistent ones.
def resample_series(x1, y1, x2, y2, freq=100):
    from numpy import floor, ceil, arange

    if len(x1) != len(y1) or len(x2) != len(y2):
        print('Unequal lengths.')
        return -1

    period = 1/freq
    real_start_time = min(x1[0], x2[0])
    start_time = floor(real_start_time / period) * period
    real_final_time = max(x1[-1], x2[-1])
    final_time = ceil(real_final_time / period) * period

    time = arange(start_time, final_time, period)

    y1_i = 0
    y2_i = 0
    y1_out = []
    y2_out = []

    for t in time:
        y1_out.append(y1[y1_i])
        y2_out.append(y2[y2_i])

        while (t + period) > x1[y1_i] > t and y1_i < (len(x1) - 1):
            y1_i += 1
        while (t + period) > x2[y2_i] > t and y2_i < (len(x2) - 1):
            y2_i += 1

    return [time, y1_out, y2_out]

    # from numpy import zeros
    # x = x1 + x2
    # x.sort()
    # y_1 = zeros(len(x))
    # y_2 = zeros(len(x))
    # j = 0
    # j_max = len(y1)
    # for i in range(len(x)):
    #     if x[i] == x1[j]:
    #         y_1[i] = y1[j]
    #         j += 1
    #         if j == j_max:
    #             break
    #     else: # TODO: improve interpolation method
    #         if j > 0:
    #             y_1[i] = y1[j-1]
    # j = 0
    # j_max = len(y2)
    # for i in range(len(x)):
    #     if x[i] == x2[j]:
    #         y_2[i] = y2[j]
    #         j += 1
    #         if j == j_max:
    #             break
    #     else: # TODO: improve interpolation method
    #         if j > 0:
    #             y_2[i] = y2[j-1]
    # if crop > 0:
    #     x = x[crop:-crop]
    #     y_1 = y_1[crop:-crop]
    #     y_2 = y_2[crop:-crop]
    return [x, y_1, y_2]


def div_filter(data, factor):
    out = []
    for i in range(0, len(data), factor):
        out.append(data[i])
    return out

def calculate_accel(acc_x, acc_y, acc_z, i):
    import numpy as np
    out = np.sqrt(np.power(acc_x[i], 2) + np.power(acc_y[i], 2) + np.power(acc_z[i], 2))
    return out

def correct_fes_input(button_timestamp, button_state):
    wrong_descend = 0
    for i in range(1, len(button_state)):
        if button_state[i] == 0 and button_state[i-1] == 1:
            wrong_descend = i
        if button_state[i] == 1 and button_state[i-1] == 0 and wrong_descend != 0:
            for j in range(wrong_descend, i):
                button_state[j] = 1
            wrong_descend = 0
    return button_state

def find_classes_and_transitions(labels, time, lower_time, upper_time):
    classes = []
    transitions = []
    previous_label = []
    for label, t in zip(labels, time):
        if lower_time < t < upper_time:
            if label not in classes:
                classes.append(label)
            if label != previous_label:
                if [previous_label, label] not in transitions:
                    transitions.append([previous_label, label])
            previous_label = label

    return classes, transitions[1:]

def make_quaternions(imu):
    q = []
    # if number_of_points == 0:
    #     starting_point = 0
    # else:
    #     starting_point = len(imu.resampled_w) - number_of_points
    for i in range(len(imu.resampled_x)):
        try:
            q.append(Quaternion(imu.resampled_w[i],
                                imu.resampled_x[i],
                                imu.resampled_y[i],
                                imu.resampled_z[i]
                                ))
        except Exception:
            # Out of sync
            # print(len(imu.resampled_w))
            # print(len(imu.resampled_x))
            # print(len(imu.resampled_y))
            # print(len(imu.resampled_z))
            # print(i)
            pass
    return q

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


def generate_imu_data(imus, imu_forearm_id, imu_arm_id):
    if imus[0].id == imu_forearm_id:
        imu_0 = 0
        imu_1 = 1
    else:
        imu_1 = 0
        imu_0 = 1

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


    q0 = make_quaternions(imus[imu_0])
    q1 = make_quaternions(imus[imu_1])

    q = []
    [q.append(i * j.conjugate) for i, j in zip(q0, q1)]

    qx = []
    qy = []
    qz = []
    qw = []
    qang = []
    acc_x_0 = [i for i in imus[imu_0].resampled_acc_x]
    acc_y_0 = [i for i in imus[imu_0].resampled_acc_y]
    acc_z_0 = [i for i in imus[imu_0].resampled_acc_z]
    acc_x_1 = [i for i in imus[imu_1].resampled_acc_x]
    acc_y_1 = [i for i in imus[imu_1].resampled_acc_y]
    acc_z_1 = [i for i in imus[imu_1].resampled_acc_z]

    acc_0 = [acc_x_0, acc_y_0, acc_z_0]
    acc_1 = [acc_x_1, acc_y_1, acc_z_1]

    acc = [acc_0, acc_1]

    for quat in q:
        qw.append(quat.elements[0])
        qx.append(quat.elements[1])
        qy.append(quat.elements[2])
        qz.append(quat.elements[3])
        qang.append(angle(quat))

    dqang = np.append([0], np.diff(qang) / np.diff(t))

    return qang, dqang, acc, t
