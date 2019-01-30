# this is the server
import time
from multiprocessing.connection import Listener
# import socket
import multiprocessing
import signal
import sys
import datetime

# TODO include ctrl+C catching in all scripts

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
    now = time.time()

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
    except:
        print('Exception')
        now = datetime.datetime.now()
        filename = now.strftime('%Y%m%d%H%M') + '_' + source + '_data.txt'
        f = open(filename, 'w+')
        # server_timestamp, client_timestamp, msg
        # if IMU, msg = id, quaternions
        # if buttons, msg = state, current
        [f.write(str(i)[1:-1].replace('[','').replace(']','')+'\n') for i in server_data]
        f.close()


def server(address):
    serv = Listener(address)
    # s = socket.socket()
    # s.bind(address)
    # s.listen()
    while True:
        # client, addr = s.accept()
        client = serv.accept()
        print('Connected to {}'.format(serv.last_accepted))
        source = client.recv()

        p = multiprocessing.Process(target=do_stuff, args=(client, source))
        # do_stuff(client, addr)
        p.start()

        # signal.pause()


if __name__ == '__main__':
    server(('localhost', 5000))
