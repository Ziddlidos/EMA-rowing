import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data_processing import GetFilesToLoad
from PyQt5.QtWidgets import QApplication
from data_classification import *
import sys

# Choose file
app = QApplication(sys.argv)
source_file = GetFilesToLoad()
app.processEvents()
filename = source_file.filename[0][0]

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

[low, zero, up] = classify_by_buttons(buttons_timestamp,
                                      buttons_values,
                                      imus[2].timestamp,
                                      imus[2].euler_z)

# [imu_2_z_low_timestamp,
#  imu_2_z_low_values,
#  imu_2_z_zero_timestamp,
#  imu_2_z_zero_values,
#  imu_2_z_up_timestamp,
#  imu_2_z_up_values] = separate_by_classification(imus[2].timestamp,
#                                                  imus[2].euler_z,
#                                                  processes_imu_2_z_classification)

# Plot
normal_plot = True
dash_plot = False


if normal_plot:


    plt.step(buttons_timestamp, buttons_values, 'k', label='buttons')
    # plt.plot(imus[1].timestamp, imus[1].euler_x, 'b-')
    # plt.plot(imus[1].timestamp, imus[1].euler_y, 'b:')
    # plt.plot(imus[1].timestamp, imus[1].euler_z, 'b--')
    # plt.plot(imus[2].timestamp, imus[2].euler_x, 'g-')
    # plt.plot(imus[2].timestamp, imus[2].euler_y, 'g:')
    plt.plot(imus[2].timestamp, imus[2].euler_z, 'c--', label='IMU 2 z')
    [plt.plot(packet.timestamp, packet.values, 'b.', label='Flexion') for packet in low]
    [plt.plot(packet.timestamp, packet.values, 'g.', label='Stop') for packet in zero]
    [plt.plot(packet.timestamp, packet.values, 'r.', label='Extension') for packet in up]
    # plt.plot(imus[0].timestamp, imus[0].euler_x, 'r-')
    # plt.plot(imus[0].timestamp, imus[0].euler_y, 'r:')
    # plt.plot(imus[0].timestamp, imus[0].euler_z, 'r--')
    # plt.plot(emg_1_timestamp, emg_1_values, 'm-')
    # plt.plot(emg_2_timestamp, emg_2_values, 'm:')
    # plt.plot(imu_2_z_up_timestamp, imu_2_z_up_values, 'r.', label='extension')
    # plt.plot(imu_2_z_zero_timestamp, imu_2_z_zero_values, 'g.', label='stop')
    # plt.plot(imu_2_z_low_timestamp, imu_2_z_low_values, 'b.', label='flexion')

    plt.title(filename)
    legend_elements = [Line2D([0], [0], color='b', label = 'Flexion', marker='o'),
                      Line2D([0], [0], color='g', label='Stop', marker='o'),
                      Line2D([0], [0], color='r', label='Extension', marker='o')]
    plt.legend(handles=legend_elements)

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
