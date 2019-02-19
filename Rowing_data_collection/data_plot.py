import matplotlib.pyplot as plt
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data_processing import GetFilesToLoad, resample_series
from PyQt5.QtWidgets import QApplication
from data_classification import *
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy.signal import medfilt
import logging

sys.stdout = open('results.txt', 'w')

# Choose file
# app = QApplication(sys.argv)
# source_file = GetFilesToLoad()
# app.processEvents()
# filename = source_file.filename[0][0]
filename = 'Estevao_rowing.out'

plt.rcParams['svg.fonttype'] = 'none'
logging.basicConfig(filename='results.txt', level=logging.DEBUG)

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

print('Resampling and synchronizing...')
[t, imus[2].resampled_euler_z, imus[0].resampled_euler_z] = resample_series(imus[2].timestamp,
                                                                            imus[2].euler_z,
                                                                            imus[0].timestamp,
                                                                            imus[0].euler_z)
[t, imus[2].resampled_euler_x, imus[0].resampled_euler_x] = resample_series(imus[2].timestamp,
                                                                            imus[2].euler_x,
                                                                            imus[0].timestamp,
                                                                            imus[0].euler_x)
[t, imus[2].resampled_euler_y, imus[0].resampled_euler_y] = resample_series(imus[2].timestamp,
                                                                            imus[2].euler_y,
                                                                            imus[0].timestamp,
                                                                            imus[0].euler_y)

[low, zero, up] = classify_by_buttons(buttons_timestamp, buttons_values, imus[2].timestamp, imus[2].euler_z)

classification0 = classify_by_buttons_in_order(buttons_timestamp, buttons_values, t)

dz0 = np.append([0], np.diff(imus[0].resampled_euler_z)/np.diff(t))
dz2 = np.append([0], np.diff(imus[2].resampled_euler_z)/np.diff(t))
dx0 = np.append([0], np.diff(imus[0].resampled_euler_x)/np.diff(t))
dx2 = np.append([0], np.diff(imus[2].resampled_euler_x)/np.diff(t))
dy0 = np.append([0], np.diff(imus[0].resampled_euler_y)/np.diff(t))
dy2 = np.append([0], np.diff(imus[2].resampled_euler_y)/np.diff(t))


def find_transitions(timestamp, values):
    transitions_times = []
    transitions_values = []
    for i in range(1, len(timestamp)):
        if values[i] != values[i-1]:
            transitions_times.append(timestamp[i])
            transitions_values.append(values[i])
    return transitions_times, transitions_values

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


normal_plot = False
dash_plot = False

number_of_points = 10
filter_size = 11

training_lower_time_table = [200, 300, 400, 500, 600]
training_upper_time_table = [275, 375, 475, 575, 675]
testing_lower_time_table = training_upper_time_table
testing_upper_time_table = [300, 400, 500, 600, 700]
tolerance_table = [0.1, 0.2, 0.3, 0.4, 0.5]
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

        for i in range(number_of_points, len(t)):
            if training_lower_time < t[i] < training_upper_time:
                this = []
                this += [j for j in imus[0].resampled_euler_z[i - number_of_points:i]]
                this += [j for j in imus[2].resampled_euler_z[i - number_of_points:i]]
                this += [j for j in imus[0].resampled_euler_x[i - number_of_points:i]]
                this += [j for j in imus[2].resampled_euler_x[i - number_of_points:i]]
                this += [j for j in imus[0].resampled_euler_y[i - number_of_points:i]]
                this += [j for j in imus[2].resampled_euler_y[i - number_of_points:i]]
                this += [dz0[i]]
                this += [dz2[i]]
                this += [dx0[i]]
                this += [dx2[i]]
                this += [dy0[i]]
                this += [dy2[i]]
                X.append(this)

                y.append(classification0[i])



        out_z_0 = [np.array(imus[0].resampled_euler_z[:-number_of_points])]
        out_z_2 = [np.array(imus[2].resampled_euler_z[:-number_of_points])]
        out_x_0 = [np.array(imus[0].resampled_euler_x[:-number_of_points])]
        out_x_2 = [np.array(imus[2].resampled_euler_x[:-number_of_points])]
        out_y_0 = [np.array(imus[0].resampled_euler_y[:-number_of_points])]
        out_y_2 = [np.array(imus[2].resampled_euler_y[:-number_of_points])]
        if number_of_points > 1:
            for i in range(1, number_of_points):
                out_z_0 = np.append(out_z_0, [np.array(imus[0].resampled_euler_z[i:-number_of_points + i])], 0)
                out_z_2 = np.append(out_z_2, [np.array(imus[2].resampled_euler_z[i:-number_of_points + i])], 0)
                out_x_0 = np.append(out_x_0, [np.array(imus[0].resampled_euler_x[i:-number_of_points + i])], 0)
                out_x_2 = np.append(out_x_2, [np.array(imus[2].resampled_euler_x[i:-number_of_points + i])], 0)
                out_y_0 = np.append(out_y_0, [np.array(imus[0].resampled_euler_y[i:-number_of_points + i])], 0)
                out_y_2 = np.append(out_y_2, [np.array(imus[2].resampled_euler_y[i:-number_of_points + i])], 0)
        out = np.append(out_z_0, out_z_2, 0)
        out = np.append(out, out_x_0, 0)
        out = np.append(out, out_x_2, 0)
        out = np.append(out, out_y_0, 0)
        out = np.append(out, out_y_2, 0)
        out = np.append(out, [dz0[number_of_points:]], 0)
        out = np.append(out, [dz2[number_of_points:]], 0)
        out = np.append(out, [dx0[number_of_points:]], 0)
        out = np.append(out, [dx2[number_of_points:]], 0)
        out = np.append(out, [dy0[number_of_points:]], 0)
        out = np.append(out, [dy2[number_of_points:]], 0)
        out = list(out.T)


        classifier = LinearDiscriminantAnalysis()
        classifier.fit(X, y)
        predicted_values = classifier.predict(out)
        predicted_values = medfilt(predicted_values, filter_size)


        print('Evaluating...')
        evaluated_buttons_timestamp = []
        evaluated_buttons_values = []
        evaluated_predicted_time = []
        evaluated_predicted_values = []
        for i in range(len(buttons_timestamp)):
            if testing_lower_time < buttons_timestamp[i] < testing_upper_time:
                evaluated_buttons_timestamp.append(buttons_timestamp[i])
                evaluated_buttons_values.append(buttons_values[i])
        for i in range(len(t)):
            if testing_lower_time < t[i] < testing_upper_time:
                evaluated_predicted_time.append(t[i])
                evaluated_predicted_values.append(predicted_values[i])

        [real_transitions_times, real_transitions_values] = find_transitions(evaluated_buttons_timestamp,
                                                                             evaluated_buttons_values)
        [predicted_transitions_times, predicted_transitions_values] = find_transitions(evaluated_predicted_time,
                                                                                       evaluated_predicted_values)

        [performance, error, false_transitions, false_predicted_time, false_predicted_values] = calculate_performance(real_transitions_times,
                                                     real_transitions_values,
                                                     predicted_transitions_times,
                                                     predicted_transitions_values,
                                                     tolerance)
        total_error[int(tolerance * 10)-1].append(error)
        # plt.figure()
        # plt.hist(error, bins=10)
        # plt.title('Time frame: {}s-{}s. Tolerance: {}s.'.format(training_lower_time, testing_upper_time, tolerance))
        # plt.savefig('{}s-{}s_tolerance_{}.svg'.format(training_lower_time, testing_upper_time, tolerance))
        # plt.show()
        performance = 0
        total = 0
        for i in range(1, len(t)):
            if testing_lower_time < t[i] < testing_upper_time:
                if predicted_values[i] == classification0[i]:
                    performance += 1
                total += 1
        print('Performance point-by-point: {}%'.format(np.round(performance/total*100, 2)))
        print('##########################################################################################\n\n\n\n')


        # Plots
        if normal_plot:
            print('Plotting...')
            print('IMU 0: {}'.format(imus[0].id))
            print('IMU 1: {}'.format(imus[1].id))
            print('IMU 2: {}'.format(imus[2].id))
            plt.step(buttons_timestamp, buttons_values, 'k', label='buttons')
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
            plt.plot(t[number_of_points:], predicted_values, 'g:', label='Predicted')
            plt.step(false_predicted_time, false_predicted_values, 'b--', label='False predictions')
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
                if 'emg1' in input_data:
                    # emg[0] = True
                    include = [{'x': emg_1_timestamp, 'y': emg_1_values, 'name': 'emg1'}]
                    graph_data = graph_data + include
                if 'emg2' in input_data:
                    # emg[1] = True
                    include = [{'x': emg_2_timestamp, 'y': emg_2_values, 'name': 'emg2'}]
                    graph_data = graph_data + include

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

for i in range(5):
    plt.figure()
    plt.hist(total_error[i], bins=10)
    plt.title('Total error for tolerance = {}s'.format((i + 1) / 10))
    plt.savefig('total_error_hist_tolerance_{}s.svg'.format((i + 1) / 10))

plt.show()
