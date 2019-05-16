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
from data_processing import *
from PyQt5.QtWidgets import QApplication
from data_classification import *
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import numpy as np
from scipy.signal import medfilt
import logging
from matplotlib.patches import Ellipse

# mode = 'singleLDA'
mode = 'switchingLDA'
# mode = 'manual'

simulate_with_different_data = False

normal_plot = True
# dash_plot = False

# number_of_points = 149
window_size = 0.5
if mode == 'singleLDA':
    confidence_level = [0.85]
else:
    confidence_level = [0.5, 0.5, 0.5]

# accel filter
filter_acc = True
# cutoff = 0.5
fs = 50
filter_size = 3
output_command_filter_size = 3
number_of_stds = 1


imu_forearm_id = 4
imu_arm_id = 5


initial_time = 60
total_time = 150

accel_threshold = 0.05

# classes = [-1, 1, 0]


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
# filename = 'Data/lucas_with_accel_01.out'
# filename = 'Data/roberto_03.out'

plt.rcParams['svg.fonttype'] = 'none'
logging.basicConfig(filename='Data/results.txt', level=logging.DEBUG)

data = {}

# Load data
print('Loading data from file {}'.format(filename))
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

qang, dqang, acc, t = generate_imu_data(imus, imu_forearm_id, imu_arm_id)

avg_f = round(len(t) / (t[-1] - t[0]))
print('Average frequency: {}'.format(avg_f))
number_of_points = int(round(avg_f * window_size))

buttons_values = correct_fes_input(buttons_timestamp, buttons_values)


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
dqang_last_zero = []
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

acc_x_0_filtered = medfilt(acc[0][0], filter_size)
acc_y_0_filtered = medfilt(acc[0][1], filter_size)
acc_z_0_filtered = medfilt(acc[0][2], filter_size)
acc_x_1_filtered = medfilt(acc[1][0], filter_size)
acc_y_1_filtered = medfilt(acc[1][1], filter_size)
acc_z_1_filtered = medfilt(acc[1][2], filter_size)


# sys.exit()

###############################################################################################
###############################################################################################
# Data plot
###############################################################################################
###############################################################################################

def plot_data():

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

    # plt.figure('Feature crossing')
    # plt.title('Angle avg x diff')
    # plt.plot(qang_avg_low[0:round(len(qang_low)/div_factor)], dqang_last_low[0:round(len(qang_low)/div_factor)], 'b.')
    # plt.plot(qang_avg_zero[0:round(len(qang_zero)/div_factor)], dqang_last_zero[0:round(len(qang_zero)/div_factor)], 'k.')
    # plt.plot(qang_avg_up[0:round(len(qang_up)/div_factor)], dqang_last_up[0:round(len(qang_up)/div_factor)], 'r.')

    # plt.show()
    # quit()
plot_data()

###############################################################################################
###############################################################################################
# Machine learning
###############################################################################################
###############################################################################################

# [low, zero, up] = classify_by_buttons(buttons_timestamp, buttons_values, imus[2].timestamp, imus[2].euler_z)

classification0 = classify_by_buttons_in_order(buttons_timestamp, buttons_values, t)

def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        for piece_of_data in data:
            pickle.dump(piece_of_data, f)


training_lower_time_table = [initial_time]  # [200, 300, 400, 500, 600]
training_upper_time_table = [total_time]  # [round(total_time * 3 / 4)] # [275, 375, 475, 575, 675]
testing_lower_time_table = training_lower_time_table  # training_upper_time_table
testing_upper_time_table = [total_time]  # [300, 400, 500, 600, 700]


trial = 0
training_lower_time = training_lower_time_table[trial]
training_upper_time = training_upper_time_table[trial]
testing_lower_time = testing_lower_time_table[trial]
testing_upper_time = testing_upper_time_table[trial]
total_length = len(classification0)

# training
print('Training...')
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

# saving trained LDAs and evaluating data
target_file = GetFileToSave()
print('Saving classifier to file {}'.format(target_file.filename[0]))
save_to_file([lda, classes, trasitions, window_size, avg_f, confidence_level], target_file.filename[0])


###############################################################################################
###############################################################################################
# Simulation
###############################################################################################
###############################################################################################


print('Generating evaluation data...')

if simulate_with_different_data:
    # Choose file
    # app = QApplication(sys.argv)
    source_file = GetFilesToLoad()
    app.processEvents()
    filename = source_file.filename[0][0]

    print('Data file to simulate: {}'.format(filename))

    # imus_sim = parse_imus_file(filename, 1557238771.5498023)
    imus_sim = parse_imus_file(filename)

    qang_sim, dqang_sim, acc_sim, t_sim = generate_imu_data(imus_sim, imu_forearm_id, imu_arm_id)
else:
    qang_sim = qang
    dqang_sim = dqang
    acc_sim = acc
    t_sim = t


if filter_size > number_of_points:
    filter_size = number_of_points
out = []
if number_of_points > 1:
    for i in range(0, len(qang_sim) - number_of_points):
        out.append([
            np.mean(qang_sim[i:number_of_points + i]),
            np.mean(dqang_sim[i:number_of_points + i]),
            np.mean(medfilt(acc_sim[0][0][i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_sim[0][1][i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_sim[0][2][i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_sim[1][0][i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_sim[1][1][i:number_of_points + i], filter_size)),
            np.mean(medfilt(acc_sim[1][2][i:number_of_points + i], filter_size))
        ])

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

###############################################################################################
###############################################################################################
# Evaluation
###############################################################################################
###############################################################################################

def evaluate_performance():
    temp_t = []
    temp_prediction = []
    temp_truth = []
    temp_score = []

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

    return temp_t, temp_truth, temp_prediction, temp_score


if not simulate_with_different_data:
    temp_t, temp_truth, temp_prediction, temp_score = evaluate_performance()

###############################################################################################
###############################################################################################
# Result plotting
###############################################################################################
###############################################################################################

if normal_plot:
    print('Plotting...')

    if mode == 'switchingLDA' and not simulate_with_different_data:
        # plt.figure()
        fig, ax = plt.subplots(len(classes), 1, sharex=True, sharey=True)
        fig.canvas.set_window_title('Each LDA performance')
        # print('IMU 2: {}'.format(imus[2].id))
        plt.step(buttons_timestamp, buttons_values, 'k', label='FES')
        for i in range(len(classes)):
            ax[i].step(buttons_timestamp, buttons_values, 'k', label='FES')
            ax[i].step(t[number_of_points:], [prediction[i] for prediction in predictions])
            plt.title('LDA {}'.format(i))

    if simulate_with_different_data:
        # fig = plt.figure('Comparison between original and new data')
        # plt.plot(t, qang, label='original')
        # plt.plot(t_sim, qang_sim, label='new data')
        # plt.legend()

        fig, ax1 = plt.subplots(1, 1, sharex=True)
        fig.canvas.set_window_title('Simulation result')
        ax1.plot(t_sim, qang_sim, label='angle')
        ax1.set_ylabel('Degrees')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.step(t_sim[number_of_points:], output_command, color='C1', label='prediction')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_ylabel('Flex=-1, Off=0, Ext=1')
        ax1.legend()
    else:
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        fig.canvas.set_window_title('Simulation result')
        ax1.plot(t, qang, color='C0', label='angle')
        ax1.set_ylabel('Degrees')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.step(temp_t, temp_truth, color='k', label='truth')
        ax2.step(temp_t, temp_prediction, color='C1', label='prediction')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_ylabel('Flex=-1, Off=0, Ext=1')
        ax1.legend()
        ax1.set_title(mode)


    # fig = plt.figure('Frequency analysis')
    # plt.plot(t[1:], 1/np.diff(t))
    # plt.plot(t[1:], medfilt(1/np.diff(t), 25))
    # plt.ylim([-10, 150])
    # print('Average frequency: {}'.format(len(t)/(t[-1]-t[1])))

    plt.show()
