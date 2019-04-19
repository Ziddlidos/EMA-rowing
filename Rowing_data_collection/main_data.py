"""
This scripts open socket connections to receive data from other programs and save them with a unified timestamp.
It is useful for syncing data from different sources.
It has a real time plot if desired that must be customized if used.
Right now it is set to open:
- python processing socket connection to receive data from other python scripts, particularly IMU data
- 2 regular TCPIP socket connections to receive data from non python programs, particularly EMG data from MATLAB.
Author: Lucas Fonseca
Contact: lucasafonseca@lara.unb.br
Date: Feb 25th 2019
Last update: April 12th 2019
"""

# this is the server
import time
from multiprocessing.connection import Listener
import socket
import multiprocessing
import sys
import datetime
import struct
import math
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyquaternion import Quaternion

import json

real_time_plot = True

# x and y are used to graph results in real time
size_of_graph = 10000
t = multiprocessing.Array('d', size_of_graph)
ang = multiprocessing.Array('d', size_of_graph)
fes = multiprocessing.Array('d', size_of_graph)
running = multiprocessing.Value('b')
start_time = time.time()
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


        curve_x.setData(t[-size_of_graph:-1], ang[-size_of_graph:-1])
        curve_y.setData(t[-size_of_graph:-1], fes[-size_of_graph:-1])
        # ptr = 0
        # if ptr == 0:
        #     my_plot.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
        # ptr += 1


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(15)


def on_exit(sig, frame):
    # global imu_data
    # print(imu_data)
    # now = datetime.datetime.now()
    # filename = now.strftime('%Y%m%d%H%M') + '_IMU_data.txt'
    # f = open(filename, 'w+')
    # [f.write(i) for i in imu_data]
    # f.close()
    print('Good bye')
    sys.exit(0)


def calculate_distance(q0, q1):

    # I think this one works
    # new_angle = 2 * Quaternion.distance(q0, q1) * 180 / math.pi
    #
    # if q0.elements[2] >= q1.elements[2]:
    #     signal = 1
    # else:
    #     signal = -1
    #
    # if new_angle > 180:
    #     new_angle = 360 - new_angle
    # new_angle = new_angle * signal

    # This one works!
    # new_angle = q0.conjugate * q1
    # new_angle = 2 * math.atan2(math.sqrt(math.pow(new_angle.vector[0], 2) +
    #                                      math.pow(new_angle.vector[1], 2) +
    #                                      math.pow(new_angle.vector[2], 2)),
    #                            new_angle.real)
    #
    # new_angle = new_angle * 180 / math.pi
    # if new_angle > 180:
    #     new_angle = 360 - new_angle

    # The best!!!
    try:
        new_quat = q0 * q1.conjugate
        qr = new_quat.elements[0]
        if qr > 1:
            qr = 1
        elif qr < -1:
            qr = -1
        new_angle = 2 * math.acos(qr)
        new_angle = new_angle * 180 / math.pi
        if new_angle > 180:
            new_angle = 360 - new_angle

    except Exception as e:
        print('Error calculation angle: ' + str(e) + 'on line ' + str(sys.exc_info()[2].tb_lineno))
        print('qr = {}'.format(new_quat.elements[0]))

    return new_angle


def do_stuff(client, source, t, ang, fes, start_time, running, imu_data):

    def save_data():
        now = datetime.datetime.now()
        filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
        f = open(filename, 'w+')
        # server_timestamp, client_timestamp, msg
        # if IMU, msg = id, quaternions
        # if buttons, msg = state, current
        [f.write(str(i)[1:-1].replace('[', '').replace(']', '') + '\r\n') for i in server_data]
        f.close()

    imu1 = [Quaternion(1, 0, 0, 0)]
    imu2 = [Quaternion(1, 0, 0, 0)]
    def update_plot():
        global t, ang, fes, start_time

        t[0:-1] = t[1:]
        t[-1] = time.time() - start_time

        # print(source)
        if source == 'imus':
            new_quat = Quaternion(data[2], data[3], data[4], data[5])
            id = data[1]

            if id == imu1_id:
                imu1.append(new_quat)
            elif id == imu2_id:
                imu2.append(new_quat)
            else:
                # print('ID not known')
                return

            new_angle = calculate_distance(imu1[-1], imu2[-1])

            ang[0:-1] = ang[1:]
            ang[-1] = new_angle

            fes[0:-1] = fes[1:]
            fes[-1] = fes[-2]


        elif source == 'stim':
            # print('update stim')
            stim_state = data[1]
            state = 0
            if stim_state == 'stop':
                state = 0
            elif stim_state == 'extension':
                state = 1
            elif stim_state == 'flexion':
                state = -1
            fes[0:-1] = fes[1:]
            fes[-1] = state * 45 + 45

            ang[0:-1] = ang[1:]
            ang[-1] = ang[-2]

    server_data = []

    try:
        while running.value:
            data = client.recv()
            if not data == '':
                server_data.append([time.time(), data])
                imu_data[data[1]] = data[:]+['|']
                # print('received stim data')
                if real_time_plot:
                    update_plot()

        save_data()

    except Exception as e:
        print('Exception raised: ', str(e), ', on line ', str(sys.exc_info()[2].tb_lineno))
        print('Connection  to {} closed'.format(source))
        save_data()


def do_stuff_socket(client, source, x, channel):
    # global x
    server_data = []
    # signal.signal(signal.SIGINT, on_exit)
    # now = time.time()

    try:
        while True:
            # print(address)
            # time.sleep(1)
            data = client.recv(4096)
            if len(data) > 2:
                # print('Data: ', data)
                # print('Data length: ', len(data))
                number_of_packets = math.floor(len(data) / 8)
                # print(number_of_packets)
                packets = []
                for i in range(number_of_packets):
                    this_packet = float(struct.unpack('!d', data[i * 8:i * 8 + 8])[0])
                    packets.append(this_packet)
                    # print('Double: ', this_packet)
                # for i in range(len(data)-3):
                #     print('Float from byte ', i, ': ', struct.unpack('!f', data[i:i+4]))
                # print('First 4 bytes unpacked: ',struct.unpack('!f', data[0:4]))
                # print('Last 4 bytes unpacked: ', struct.unpack('!f', data[-4:]))
                # print('')
                # x.append(packets)
                if channel == 1:
                    x[0:-number_of_packets] = x[number_of_packets:]
                    x[-number_of_packets:] = packets
                elif channel == 2:
                    y[0:-number_of_packets] = y[number_of_packets:]
                    y[-number_of_packets:] = packets
                server_data.append([time.time(), packets])
                # print(packets[0])
                # print(imu_data)
                # print(source, data)
                # time.sleep(1)
            else:
                raise ValueError('Connection closed')
            # print(1/(time.time()-now))
            # now = time.time()
    except Exception as e:
        print('Exception raised: ', str(e))
        print('Connection  to {} closed'.format(source))
        now = datetime.datetime.now()
        filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_ch' + str(channel) + '_data.txt'
        f = open(filename, 'w+')
        # server_timestamp, client_timestamp, msg
        # if IMU, msg = id, quaternions
        # if buttons, msg = state, current
        [f.write(str(i)[1:-1].replace('[', '').replace(']', '') + '\r\n') for i in server_data]
        f.close()


def server(address, port, t, ang, fes, start_time, running, imu_data):
    serv = Listener((address, port))
    # s = socket.socket()
    # s.bind(address)
    # s.listen()
    while running.value:
        # print(running.value)
        # client, addr = s.accept()
        client = serv.accept()
        print('Connected to {}'.format(serv.last_accepted))
        source = client.recv()
        print('Source: {}'.format(source))
        p = multiprocessing.Process(target=do_stuff, args=(client, source, t, ang, fes, start_time, running, imu_data))
        # do_stuff(client, addr)
        p.start()

        # signal.pause()
    print('End of process server')


def socket_server(address, port, x, channel):
    s = socket.socket()
    s.bind((address, port))
    s.listen()
    while True:
        conn, addr = s.accept()
        print('Connected to {}'.format(addr))
        time.sleep(1)
        if s:
            source = str(conn.recv(4096))[2:-1]
            if len(source) > 20:
                source = 'EMG'
        else:
            print('Disconnected from {}'.format(addr))
            break
        print('Source: {}'.format(source))
        p = multiprocessing.Process(target=do_stuff_socket, args=(conn, source, x, channel))
        # do_stuff(client, addr)
        p.start()

def vr_server(address, port, imu_data):
    # global imu_data
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((address, port))
    s.listen()
    conn, addr = s.accept()
    if s:
            print('Connected to {}'.format(addr))
            source = str(conn.recv(4096))[2:-1]
            if len(source) > 4:
                source = 'VR'
            print('Source: {}'.format(source))
    while True:
        # print('Connection attempt')
        receiveTime = str(conn.recv(4096))[2:-1]
        # print('Connection attempt 1')
        # time.sleep(1/720)
        # print('Connection attempt 2')
        if s:
            # imu_data.append([time.time(), imu_data])
            # print('Sent message: {}'.format(list(imu_data)))
            # imu_data[:] = [float(receiveTime), imu_data[:]]
            imu_data['velocity'] = '1.0|' # TODO: real value goes here
            out_data = json.dumps(receiveTime + '|') + json.dumps(dict(imu_data))
            conn.send(out_data.encode())
            # del imu_data[:]
            # print(out_data)
        else:
            print('Disconnected from {}'.format(addr))
            break

manager = multiprocessing.Manager()
imu_data = manager.dict()
# imu_json = multiprocessing.Value()

if __name__ == '__main__':
    # mserver = multiprocessing.Process(target=server, args=('', 50001, imu_data))
    mserver = multiprocessing.Process(target=server, args=('', 50001, t, ang, fes, start_time, running, imu_data))
    # mserver = threading.Thread(target=server, args=(('', 50001),))
    running.value = 1
    mserver.start()

    # sserver = multiprocessing.Process(target=socket_server, args=('', 50002, x, 1))
    # sserver.start()
    # sserver2 = multiprocessing.Process(target=socket_server, args=('', 50003, x, 2))
    # sserver2.start()
    # sserver = multiprocessing.Process(target=socket_server, args=('', 50002, x, 1))
    # sserver.start()
    # sserver2 = multiprocessing.Process(target=socket_server, args=('', 50003, x, 2))
    # sserver2.start()
    sserver3 = multiprocessing.Process(target=vr_server, args=('', 50004, imu_data))
    sserver3.start()
    # server(('', 50000))

    if real_time_plot:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    else:
        input('Press ENTER do finish')
    print('Good bye')
    running.value = 0
    mserver.terminate()
