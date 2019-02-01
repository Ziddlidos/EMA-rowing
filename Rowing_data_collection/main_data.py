# this is the server
import time
from multiprocessing.connection import Listener
import socket
import multiprocessing
# import threading
# import signal
import sys
import datetime
import struct
import math
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

# TODO include ctrl+C catching in all scripts

app = QtGui.QApplication([])
# app.aboutToQuit.connect(on_exit)
win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')
pg.setConfigOptions(antialias=True)
my_plot = win.addPlot(title="Updating plot")
size_of_graph = 10000
my_plot.setRange(xRange=(0,size_of_graph), yRange=(-50,50))
my_plot.enableAutoRange('xy', False)

curve_x = my_plot.plot(pen='b') # EMG

x = multiprocessing.Array('d', size_of_graph)

for i in range(size_of_graph):
    x[i] = 0
# x = [0] * 1000
# ptr = 0
def update():
    global curve_x, x, ptr, p6
    # print('New data: ', x[-1])
    curve_x.setData(x[-size_of_graph:-1])
    # ptr = 0
    # if ptr == 0:
    #     my_plot.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    # ptr += 1
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(15)

# imu_data = []

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

def do_stuff(client, source):
    server_data = []
    # signal.signal(signal.SIGINT, on_exit)
    # now = time.time()
    number_of_packets = 1
    try:
        while True:
            # print(address)
            # time.sleep(1)
            data = client.recv()
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
        [f.write(str(i)[1:-1].replace('[','').replace(']','')+'\r\n') for i in server_data]
        f.close()

def do_stuff_socket(client, source, x):
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
                number_of_packets = math.floor(len(data)/8)
                packets = []
                for i in range(number_of_packets):
                    this_packet = float(struct.unpack('!d', data[i*8:i*8+8])[0])
                    packets.append(this_packet)
                    # print('Double: ', this_packet)
                # for i in range(len(data)-3):
                #     print('Float from byte ', i, ': ', struct.unpack('!f', data[i:i+4]))
                # print('First 4 bytes unpacked: ',struct.unpack('!f', data[0:4]))
                # print('Last 4 bytes unpacked: ', struct.unpack('!f', data[-4:]))
                # print('')
                # x.append(packets)
                x[0:-number_of_packets] = x[number_of_packets:]
                x[-number_of_packets:] = packets
                server_data.append([time.time(), packets])

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
        filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
        f = open(filename, 'w+')
        # server_timestamp, client_timestamp, msg
        # if IMU, msg = id, quaternions
        # if buttons, msg = state, current
        [f.write(str(i)[1:-1].replace('[','').replace(']','')+'\r\n') for i in server_data]
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
        p = multiprocessing.Process(target=do_stuff, args=(client, source))
        # do_stuff(client, addr)
        p.start()

        # signal.pause()


def socket_server(address, port, x):
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
        p = multiprocessing.Process(target=do_stuff_socket, args=(conn, source, x))
        # do_stuff(client, addr)
        p.start()

if __name__ == '__main__':
    mserver = multiprocessing.Process(target=server, args=('', 50001))
    # mserver = threading.Thread(target=server, args=(('', 50001),))
    mserver.start()
    sserver = multiprocessing.Process(target=socket_server, args=('', 50002, x))
    sserver.start()
    # server(('', 50000))
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
