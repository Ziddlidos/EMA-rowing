import pickle
from PyQt5.QtWidgets import QApplication
import threading
from multiprocessing.connection import Listener
# import select
import time
import datetime
from data_processing import IMU, resample_series, make_quaternions, angle, GetFilesToLoad
from data_classification import Classifier
import numpy as np
from scipy.signal import medfilt
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

# mode = 'singleLDA'
mode = 'switchingLDA'
# mode = 'manual'

# Choose file
app = QApplication(sys.argv)
source_file = GetFilesToLoad()
app.processEvents()
filename = source_file.filename[0][0]
classifier = filename

imu_forearm_id = 4
imu_arm_id = 5

imu_forearm = IMU(imu_forearm_id)
imu_arm = IMU(imu_arm_id)

# number_of_points = 50
filter_size = 3
output_filter_size = 3

######################

real_time_plot = True

# x and y are used to graph results in real time
size_of_graph = 500
t = [0 for i in range(size_of_graph)]
ang = [0 for i in range(size_of_graph)]
fes = [0 for i in range(size_of_graph)]
start_time = time.time()
time_control = [0]
if real_time_plot:

    imu1_id = 5
    imu2_id = 8
    for i in range(size_of_graph):
        t[i] = 0
        ang[i] = 0

    # global curve_x, t, curve_y, ang
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('pyqtgraph example: Plotting')
    pg.setConfigOptions(antialias=True)
    my_plot = win.addPlot(title="Updating plot")

    # my_plot.setRange(xRange=(0, size_of_graph/10), yRange=(-20, 400))
    # my_plot.enableAutoRange('xy', False)


    curve_x = my_plot.plot(x = t, pen='b')
    curve_y = my_plot.plot(x = t, pen='r')


    # x = [0] * 1000
    # ptr = 0

    def update():
        global curve_x, t, curve_y, ang
        # print('New data: ', x[-1])
        # print('updating plot')
        try:
            # print(t[-100:])
            curve_x.setData(t[-size_of_graph:-1], ang[-size_of_graph:-1])
            curve_y.setData(t[-size_of_graph:-1], fes[-size_of_graph:-1])
        except Exception as e:
            print('t: {}'.format(t))
            print('fes: {}'.format(fes))
            raise Exception
        # ptr = 0
        # if ptr == 0:
        #     my_plot.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
        # ptr += 1


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(15)

command = [0 for i in range(output_filter_size)]
angles = [0 for i in range(output_filter_size)]
timestamp = [time.time() for i in range(output_filter_size)]
running = True
# Load classifier from file
with open(classifier, 'rb') as f:
    try:
        print('Loading...')
        classifiers = pickle.load(f)
        classes = pickle.load(f)
        transitions = pickle.load(f)
        window_size = pickle.load(f)
        freq = pickle.load(f)
        confidence_level = pickle.load(f)

        # confidence_level = [0.5, 0.5, 0.5]
        # Override confidence_level if desired
        # confidence_level = [0.75, 0.75, 0.75]
        print('Machine learning file loaded: {}'.format(classifier))
        print('Classes: {}'.format(classes))
        print('Transitions: {}'.format(transitions))
        print('Window size: '.format(window_size))
        print('Freq: {}'.format(freq))
        print('Confidence level: {}'.format(confidence_level))

    except EOFError:
        print('Loading complete')


def imu_thread(client_list):
    global imu_forearm, imu_arm, running
    source = 'imu'
    client = client_list
    server_data = []
    # signal.signal(signal.SIGINT, on_exit)
    # now = time.time()
    # number_of_packets = 1
    try:
        while running:
            # print(address)
            # time.sleep(1)
            data = client.recv()
            if not data == '':
                # print(data)
                server_data.append([time.time(), data])
                timestamp_imu = data[0]
                id = data[1]
                x = data[2]
                y = data[3]
                z = data[4]
                w = data[5]
                acc_x = data[6]
                acc_y = data[7]
                acc_z = data[8]

                if id == imu_forearm.id:
                    imu_forearm.timestamp.append(timestamp_imu)
                    imu_forearm.x_values.append(x)
                    imu_forearm.y_values.append(y)
                    imu_forearm.z_values.append(z)
                    imu_forearm.w_values.append(w)
                    imu_forearm.acc_x.append(acc_x)
                    imu_forearm.acc_y.append(acc_y)
                    imu_forearm.acc_z.append(acc_z)

                elif id == imu_arm.id:
                    imu_arm.timestamp.append(timestamp_imu)
                    imu_arm.x_values.append(x)
                    imu_arm.y_values.append(y)
                    imu_arm.z_values.append(z)
                    imu_arm.w_values.append(w)
                    imu_arm.acc_x.append(acc_x)
                    imu_arm.acc_y.append(acc_y)
                    imu_arm.acc_z.append(acc_z)

                # print(data[2], data[3], data[4], data[5])
                # print(imu_data)
                # print(source, data)
                # time.sleep(1)
            # print(1/(time.time()-now))
            # now = time.time()
    except Exception as e:
        print('Exception raised: ', str(e))
        print('Connection  to {} closed'.format(source))
        now = datetime.datetime.now()
        filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
        f = open(filename, 'w+')
        # server_timestamp, client_timestamp, msg
        # if IMU, msg = id, quaternions
        # if buttons, msg = state, current
        [f.write(str(i)[1:-1].replace('[', '').replace(']', '') + '\r\n') for i in server_data]
        f.close()
        running = False


def stim_thread(client):
    global running, command
    source = 'stim' #client.recv()
    server_data = []



    # signal.signal(signal.SIGINT, on_exit)
    # now = time.time()
    # number_of_packets = 1
    try:
        while running:
            # print(address)
            # time.sleep(1)
            # print(command[-1])
            if len(command) > filter_size:
                client.send(np.median(command[-output_filter_size:]))
            else:
                client.send(command[-1])
            data = client.recv()
            # print(data)
            if not data == '':
                # print(data)
                server_data.append([time.time(), data])
                # print(imu_data)
                # print(source, data)
                # time.sleep(1)
            # print(1/(time.time()-now))
            # now = time.time()
    except Exception as e:
        print('Exception raised: ', str(e))
        print('Connection  to {} closed'.format(source))
        now = datetime.datetime.now()
        filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
        f = open(filename, 'w+')
        # server_timestamp, client_timestamp, msg
        # if IMU, msg = id, quaternions
        # if buttons, msg = state, current
        [f.write(str(i)[1:-1].replace('[', '').replace(']', '') + '\r\n') for i in server_data]
        f.close()
        running = False


def control(lda, classes, window_size, freq, confidence_level):
    # import math
    global imu_forearm, imu_arm, imu0, imu1, command, angles, t, ang, fes, filter_size, running

    source = 'control'

    q = []
    predictions = []
    probabilities = []
    state = -1
    state_prediction = [0]
    state_probability = [0]
    number_of_points = int(round(freq * window_size))
    period = 1/freq

    c = Classifier(lda)

    if filter_size > number_of_points:
        filter_size = number_of_points

    while len(imu_forearm.timestamp) < number_of_points:
        # print(len(imu0.timestamp))
        time.sleep(period)

    print('Starting control')
    while running:

        [t0, imu_forearm.resampled_x, imu_arm.resampled_x] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                imu_forearm.x_values[-number_of_points:],
                                                                                imu_arm.timestamp[-number_of_points:],
                                                                                imu_arm.x_values[-number_of_points:])
        [t0, imu_forearm.resampled_y, imu_arm.resampled_y] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                imu_forearm.y_values[-number_of_points:],
                                                                                imu_arm.timestamp[-number_of_points:],
                                                                                imu_arm.y_values[-number_of_points:])
        [t0, imu_forearm.resampled_z, imu_arm.resampled_z] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                imu_forearm.z_values[-number_of_points:],
                                                                                imu_arm.timestamp[-number_of_points:],
                                                                                imu_arm.z_values[-number_of_points:])
        [t0, imu_forearm.resampled_w, imu_arm.resampled_w] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                imu_forearm.w_values[-number_of_points:],
                                                                                imu_arm.timestamp[-number_of_points:],
                                                                                imu_arm.w_values[-number_of_points:])
        [t0, imu_forearm.resampled_acc_x, imu_arm.resampled_acc_x] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                        imu_forearm.acc_x[-number_of_points:],
                                                                                        imu_arm.timestamp[-number_of_points:],
                                                                                        imu_arm.acc_x[-number_of_points:])
        [t0, imu_forearm.resampled_acc_y, imu_arm.resampled_acc_y] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                        imu_forearm.acc_y[-number_of_points:],
                                                                                        imu_arm.timestamp[-number_of_points:],
                                                                                        imu_arm.acc_y[-number_of_points:])
        [t0, imu_forearm.resampled_acc_z, imu_arm.resampled_acc_z] = resample_series(imu_forearm.timestamp[-number_of_points:],
                                                                                        imu_forearm.acc_z[-number_of_points:],
                                                                                        imu_arm.timestamp[-number_of_points:],
                                                                                        imu_arm.acc_z[-number_of_points:])

        q0 = make_quaternions(imu_forearm)
        q1 = make_quaternions(imu_arm)

        q = []
        [q.append(i * j.conjugate) for i, j in zip(q0, q1)]

        qx = []
        qy = []
        qz = []
        qw = []
        qang = []
        acc_x_0 = [i for i in imu_forearm.resampled_acc_x]
        acc_y_0 = [i for i in imu_forearm.resampled_acc_y]
        acc_z_0 = [i for i in imu_forearm.resampled_acc_z]
        acc_x_1 = [i for i in imu_arm.resampled_acc_x]
        acc_y_1 = [i for i in imu_arm.resampled_acc_y]
        acc_z_1 = [i for i in imu_arm.resampled_acc_z]


        for quat in q:
            qw.append(quat.elements[0])
            qx.append(quat.elements[1])
            qy.append(quat.elements[2])
            qz.append(quat.elements[3])
            qang.append(angle(quat))

        # print('len q: {}'.format(len(q)))
        da = np.diff(qang)
        dt = np.diff(t0)
        while len(dt) > len(da):
            # print('correcting dt')
            dt = dt[1:]
        while len(dt) < len(da):
            # print('correcting dt')
            da = da[1:]
        dqang = np.append([0], da / dt)
        # q.append(new_q0[-1] * new_q1[-1].conjugate)
        if t0[-1] == time_control[-1]:
            # Repeated value. Ignore.
            # print('ignoring data')
            continue
        time_control.append(t0[-1])
        angles.append(angle(q[-1]))
        out = []

        out.append([
            np.mean(qang[-number_of_points:]),
            np.mean(dqang[-number_of_points:]),
            np.mean(medfilt(acc_x_0[-number_of_points:], filter_size)),
            np.mean(medfilt(acc_y_0[-number_of_points:], filter_size)),
            np.mean(medfilt(acc_z_0[-number_of_points:], filter_size)),
            np.mean(medfilt(acc_x_1[-number_of_points:], filter_size)),
            np.mean(medfilt(acc_y_1[-number_of_points:], filter_size)),
            np.mean(medfilt(acc_z_1[-number_of_points:], filter_size))
            ])

        # print(out)


        try:
            [new_prediction, new_probability] = c.classify(np.array(out).reshape(1, -1))
        except Exception:
            # print(imu0.timestamp[-number_of_points:])
            raise Exception
        predictions.append(new_prediction)
        probabilities.append(new_probability)

        if mode  == 'switchingLDA':
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

        elif mode == 'singleLDA':
            if new_probability[0] > confidence_level[0]:
                state_prediction.append(new_prediction[0])
                state_probability.append(new_probability[0])
            else:
                state_prediction.append(state_prediction[-1])
                state_probability.append(state_probability[-1])

        elif mode == 'manual':
            if state == -1 and (out[0][2] > 0.05):  # and value[5] > 0.25):
                state = 1
                state_prediction.append(state)
                state_probability.append(1)
            elif state == 1 and out[0][0] > 90 and out[0][1] < 0:
                state = 0
                state_prediction.append(state)
                state_probability.append(1)
            elif state == 0 and out[0][0] < 15:
                state = -1
                state_prediction.append(state)
                state_probability.append(1)
            else:
                state_prediction.append(state_prediction[-1])
                state_probability.append(state_probability[-1])

        result = state_prediction[-1]
        # print(result)
        timestamp.append(time.time())
        command.append(result)
        # print(command[-1])

        t[0:-1] = t[1:]
        t[-1] = time.time() - start_time

        ang[0:-1] = ang[1:]
        ang[-1] = angles[-1]
        # print(ang[-1])

        fes[0:-1] = fes[1:]
        fes[-1] = float(np.median(command[-output_filter_size:]) * 45 + 45)

    now = datetime.datetime.now()
    filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
    f = open(filename, 'w+')
    # server_timestamp, client_timestamp, msg
    # if IMU, msg = id, quaternions
    # if buttons, msg = state, current
    for i in range(len(command)):
        f.write(str(timestamp[-i]) + ', ' + str(command[i]) + '\r\n')
    # [f.write(str(i)[1:-1].replace('[', '').replace(']', '') + '\r\n') for i in server_data]
    f.close()


def server(address, port):
    global running
    serv = Listener((address, port))

    # s = socket.socket()
    # s.bind(address)
    # s.listen()
    while running:
        # client, addr = s.accept()
        # select.select([serv], [], [serv], 0.1)
        # print(wait([serv], 0.2))
        client = serv.accept()

        print('Connected to {}'.format(serv.last_accepted))
        source = client.recv()
        print('Source: {}'.format(source))
        if source == 'imus':
            t = threading.Thread(target=imu_thread, args=[client])
            t.start()
        elif source == 'stim':
            t = threading.Thread(target=stim_thread, args=([client]))
            t.start()
        # p = multiprocessing.Process(target=do_stuff, args=(client, source))
        # do_stuff(client, addr)
        # p.start()
    print('Server closed')


if __name__ == '__main__':

    control_loop = threading.Thread(target=control, args=(classifiers, classes, window_size, freq, confidence_level))
    server = threading.Thread(target=server, args=('', 50001))

    server.start()
    control_loop.start()

    if real_time_plot:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    # real_time_plot_thread = threading.Thread(target=update_plot, args=(,))
    # real_time_plot_thread.start()
    running = False

