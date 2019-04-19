import pickle
import matplotlib.pyplot as mpl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import threading
from multiprocessing.connection import Listener
import time
import datetime
from data_processing import IMU, calculate_accel
import numpy as np
from scipy.signal import medfilt
from pyquaternion import Quaternion
import math
import sys

imu_forearm_id = 4
imu_arm_id = 3

imu_forearm = IMU(imu_forearm_id)
imu_arm = IMU(imu_arm_id)

number_of_points = 50
filter_size = 29

command = [0]
timestamp = [time.time()]
running = True
# Load classifier from file
with open('Data/classifier', 'rb') as f:
    try:
        print('Loading...')
        X = pickle.load(f)
        y = pickle.load(f)
        # out = pickle.load(f)
        print('Loading complete')

    except EOFError:
        print('Loading complete')

print('Learning...')
classifier = LinearDiscriminantAnalysis()
classifier.fit(X, y)
print('Learning complete')
# predicted_values = classifier.predict(out)

# mpl.plot(predicted_values)
# mpl.show()

def imu_thread(client_list):
    global imu_forearm, imu_arm, running
    source = 'imu'
    client = client_list
    server_data = []
    # signal.signal(signal.SIGINT, on_exit)
    # now = time.time()
    # number_of_packets = 1
    try:
        while True:
            # print(address)
            # time.sleep(1)
            data = client.recv()
            if not data == '':
                # print(data)
                server_data.append([time.time(), data])
                # data = data.
                id = data[1]
                x = data[2]
                y = data[3]
                z = data[4]
                w = data[5]
                acc_x = data[6]
                acc_y = data[7]
                acc_z = data[8]

                if id == imu_forearm.id:
                    imu_forearm.x_values.append(x)
                    imu_forearm.y_values.append(y)
                    imu_forearm.z_values.append(z)
                    imu_forearm.w_values.append(w)
                    imu_forearm.acc_x.append(acc_x)
                    imu_forearm.acc_y.append(acc_y)
                    imu_forearm.acc_z.append(acc_z)
                elif id == imu_arm.id:
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
            print(command[-1])
            if len(command) > filter_size:
                client.send(np.median(command[-filter_size:]))
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


def make_quaternions(imu):
    q = []
    for i in range(len(imu.w_values)):
        q.append(Quaternion(imu.w_values[i],
                            imu.x_values[i],
                            imu.y_values[i],
                            imu.z_values[i]
                            ))
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


def control():
    global imu_forearm, imu_arm, classifier, command
    print('Starting control')
    source = 'control'
    angles = []
    q = []
    while not len(imu_arm.x_values) > number_of_points + 1 or not len(imu_forearm.x_values) > number_of_points + 1:
        pass
    while running:
        q0 = make_quaternions(imu_forearm)
        q1 = make_quaternions(imu_arm)


        q.append(q0[-1] * q1[-1].conjugate)
        angles.append(angle(q[-1]))

        if len(angles) > number_of_points:


            # out = list()
            out = []

            # out = out + angles[-number_of_points:]
            # out = out + list(np.diff(angles[-number_of_points - 1:]))

            out.append([np.mean(angles[-number_of_points:]),
                        np.diff(angles[-number_of_points:])[-1],
                        calculate_accel(imu_forearm.acc_x, imu_forearm.acc_y, imu_forearm.acc_z, -1)
                        ])

            # quaternions
            # out = out + imu_forearm.x_values[-number_of_points:]
            # out = out + imu_arm.x_values[-number_of_points:]
            # out = out + imu_forearm.y_values[-number_of_points:]
            # out = out + imu_arm.y_values[-number_of_points:]
            # out = out + imu_forearm.z_values[-number_of_points:]
            # out = out + imu_arm.z_values[-number_of_points:]
            # out = out + imu_forearm.w_values[-number_of_points:]
            # out = out + imu_arm.w_values[-number_of_points:]
            # out = out + list(np.diff(imu_forearm.x_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_arm.x_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_forearm.y_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_arm.y_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_forearm.z_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_arm.z_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_forearm.w_values[-number_of_points - 1:]))
            # out = out + list(np.diff(imu_arm.w_values[-number_of_points - 1:]))

            # print(out)
            result = classifier.predict(np.array(out).reshape(1, -1))
            timestamp.append(time.time())
            command.append(result)

            print(result)

    now = datetime.datetime.now()
    filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
    f = open(filename, 'w+')
    # server_timestamp, client_timestamp, msg
    # if IMU, msg = id, quaternions
    # if buttons, msg = state, current
    for i in range(len(command)):
        f.write(str(timestamp) + ', ' + str(command) + '\r\n')
    # [f.write(str(i)[1:-1].replace('[', '').replace(']', '') + '\r\n') for i in server_data]
    f.close()


def server(address, port):
    serv = Listener((address, port))

    # s = socket.socket()
    # s.bind(address)
    # s.listen()
    while True:
        # client, addr = s.accept()
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


if __name__ == '__main__':
    control_loop = threading.Thread(target=control)
    control_loop.start()
    server = threading.Thread(target=server, args=('', 50001))
    server.start()
    # mserver = multiprocessing.Process(target=server, args=('', 50001))
    # mserver = threading.Thread(target=server, args=(('', 50001),))
    # mserver.start()
    # sserver = multiprocessing.Process(target=socket_server, args=('', 50002, x, 1))
    # sserver.start()
    # sserver2 = multiprocessing.Process(target=socket_server, args=('', 50003, x, 2))
    # sserver2.start()
    # server(('', 50000))
    # if real_time_plot:
    #     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #         QtGui.QApplication.instance().exec_()
