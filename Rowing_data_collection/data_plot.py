import matplotlib.pyplot as plt
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

filename = 'rowing_data.out'
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

# Plot
normal_plot = False
dash_plot = True


if normal_plot:


    plt.step(buttons_timestamp, buttons_values, 'k')
    plt.plot(imus[1].timestamp, imus[1].euler_x, 'b-')
    plt.plot(imus[1].timestamp, imus[1].euler_y, 'b:')
    plt.plot(imus[1].timestamp, imus[1].euler_z, 'b--')
    plt.plot(imus[2].timestamp, imus[2].euler_x, 'g-')
    plt.plot(imus[2].timestamp, imus[2].euler_y, 'g:')
    plt.plot(imus[2].timestamp, imus[2].euler_z, 'g--')
    plt.plot(imus[0].timestamp, imus[0].euler_x, 'r-')
    plt.plot(imus[0].timestamp, imus[0].euler_y, 'r:')
    plt.plot(imus[0].timestamp, imus[0].euler_z, 'r--')
    plt.plot(emg_1_timestamp, emg_1_values, 'm-')
    plt.plot(emg_2_timestamp, emg_2_values, 'm:')
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
                {'label': 'IMU 3 - z', 'value': 'imus2z'},
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

        data = []

        if 'buttons' in input_data:
            # buttons = True
            include = [{'x': buttons_timestamp, 'y': buttons_values, 'type': 'step', 'name': 'buttons'}]
            data = data + include
        if 'imus0x' in input_data:
            # imus[0] = True
            include = [{'x': imus[0].timestamp, 'y': imus[0].x_values, 'type': 'step', 'name': 'imu0x'}]
            data = data + include
        if 'imus0y' in input_data:
            include = [{'x': imus[0].timestamp, 'y': imus[0].y_values, 'type': 'step', 'name': 'imu0y'}]
            data = data + include
        if 'imus0z' in input_data:
            include = [{'x': imus[0].timestamp, 'y': imus[0].z_values, 'type': 'step', 'name': 'imu0z'}]
            data = data + include
        if 'imus1x' in input_data:
            # imus[1] = True
            include = [{'x': imus[1].timestamp, 'y': imus[1].x_values, 'type': 'step', 'name': 'imus1x'}]
            data = data + include
        if 'imus1y' in input_data:
            include = [{'x': imus[1].timestamp, 'y': imus[1].y_values, 'type': 'step', 'name': 'imus1y'}]
            data = data + include
        if 'imus1z' in input_data:
            include = [{'x': imus[1].timestamp, 'y': imus[1].z_values, 'type': 'step', 'name': 'imus1z'}]
            data = data + include
        if 'imus2x' in input_data:
            # imus[2] = True
            include = [{'x': imus[2].timestamp, 'y': imus[2].x_values, 'type': 'step', 'name': 'imus2x'}]
            data = data + include
        if 'imus2y' in input_data:
            include = [{'x': imus[2].timestamp, 'y': imus[2].y_values, 'type': 'step', 'name': 'imus2y'}]
            data = data + include
        if 'imus2z' in input_data:
            include = [{'x': imus[2].timestamp, 'y': imus[2].z_values, 'type': 'step', 'name': 'imus2z'}]
            data = data + include
        if 'emg1' in input_data:
            # emg[0] = True
            include = [{'x': emg_1_timestamp, 'y': emg_1_values, 'type': 'step', 'name': 'emg1'}]
            data = data + include
        if 'emg2' in input_data:
            # emg[1] = True
            include = [{'x': emg_2_timestamp, 'y': emg_2_values, 'type': 'step', 'name': 'emg2'}]
            data = data + include

        return dcc.Graph(
            id='example-graph',
            figure={
                'data': data,
                'layout': {
                    'title': 'Rowing data'
                }
            },
            style={'height': 800},
        )


    app_dash.run_server(debug=False)
    # dash_process = multiprocessing.Process(target=run_dash, args=(app_dash,))
    # dash_process.start()