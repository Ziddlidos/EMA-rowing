import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QDialog
import matplotlib.pyplot as plt
from transformations import euler_from_quaternion
from numpy import linspace, mean
from plot_ui import Ui_Dialog
from PyQt5 import uic, QtWidgets

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

normal_plot = False
dash_plot = True

class plot_chooser:
    def __init__(self):

        # self.ui = uic.loadUi('plot_ui.ui')
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self.window)
        self.ui.checkBox_buttons.toggled.connect(self.update_plot)

        # self.ui.show()
        self.window.show()


    def update_plot(self):
        fig = plt.Figure()
        plt.step(buttons_timestamp, buttons_values, 'b')
        plt.show()
        print('update plot')

class GetFiles(QWidget):

    def __init__(self):
        super(GetFiles, self).__init__()
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


def separate_files(all_files):
    emg_files = [f for f in ex.filename[0] if 'EMG' in f]
    imus_files = [f for f in ex.filename[0] if 'imus' in f]
    buttons_files = [f for f in ex.filename[0] if 'buttons' in f]
    return [emg_files, imus_files, buttons_files]

def parse_button_file(filename):
    global starting_time
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

def parse_emg_file(filename):
    global starting_time
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

def parse_imus_file(filename):
    global starting_time
    lines = []
    imus = []
    imus_ids = []
    with open(filename) as inputfile:
        for line in inputfile:
            lines.append(line.split(','))
    # first_time = float(lines[0][0])
    for data in lines:
        id = float(data[2])
        if id not in imus_ids:
            imus_ids.append(id)
            imus.append(IMU(id))
        imus[imus_ids.index(id)].timestamp.append(float(data[0]) - starting_time)
        imus[imus_ids.index(id)].x_values.append(float(data[3]))
        imus[imus_ids.index(id)].y_values.append(float(data[4]))
        imus[imus_ids.index(id)].z_values.append(float(data[5]))
        imus[imus_ids.index(id)].w_values.append(float(data[6]))

    [imus[i].get_euler_angles() for i in range(len(imus))]

    return imus

def get_starting_time(filenames):
    times = []
    for filename in filenames:
        with open(filename) as inputfile:
            for line in inputfile:
                line = line.split(',')
                times.append(float(line[0]))
                break

    return min(times)

def new_plot(fig, *args):
    for i in range(0,len(args),2):
        plt.plot(args[i], args[i+1])
    plt.show()

def run_dash(app_dash):
    app_dash.run_server(debug=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GetFiles()
    
    [emg_files, imus_files, buttons_files] = separate_files(ex.filename[0])

    app.closeAllWindows()
    app.quit()

    starting_time = get_starting_time([buttons_files[0], imus_files[0], emg_files[0]])

    [buttons_timestamp, buttons_values] = [0,0]
    imus = [0]
    [emg_1_timestamp, emg_1_values] = [0,0]
    [emg_2_timestamp, emg_2_values] = [0,0]

    [buttons_timestamp, buttons_values] = parse_button_file(buttons_files[0])
    imus = parse_imus_file(imus_files[0])
    [emg_1_timestamp, emg_1_values] = parse_emg_file(emg_files[0])
    # [emg_2_timestamp, emg_2_values] = parse_emg_file(emg_files[1])
    # emg_2 = parse_emg_file(emg_files[1])

    if normal_plot:

        # fig = plt.Figure()

        # plot_process = multiprocessing.Process(target=new_plot, args=(fig, imus[1].timestamp,
        #                                                              imus[1].x_values,
        #                                                              imus[2].timestamp,
        #                                                              imus[2].x_values))
        # plot_process.start()

        # graph = f.add_subplot(111)
        # graph.step(buttons_timestamp, buttons_values, 'k')
        # graph.show()

        # ui = QDialog(Ui_Dialog)
        # window = plot_chooser()

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


        dash_process = multiprocessing.Process(target=run_dash, args=(app_dash,))
        dash_process.start()

    # sys.exit(app.quit())
