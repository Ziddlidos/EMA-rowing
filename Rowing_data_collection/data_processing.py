from PyQt5.QtWidgets import QWidget, QFileDialog
from transformations import euler_from_quaternion
from numpy import mean


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

def separate_files(filenames):
    emg_files = [f for f in filenames if 'EMG' in f]
    imus_files = [f for f in filenames if 'imus' in f]
    buttons_files = [f for f in filenames if 'buttons' in f]
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

def parse_imus_file(filename, starting_time):
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

def run_dash(app_dash):
    app_dash.run_server(debug=True)
