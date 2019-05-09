from data_processing import GetFilesToLoad
from PyQt5.QtWidgets import QApplication
from multiprocessing.connection import Client
import sys
from time import sleep

# Choose file
# app = QApplication(sys.argv)
# source_file = GetFilesToLoad()
# app.processEvents()
# filename = source_file.filename[0][0]

filename = '201905071132_imu_data.txt'

lines = []
timestamp = []
id = []
w = []
x = []
y = []
z = []
acc_x = []
acc_y = []
acc_z = []

print('Loading data...')
with open(filename) as inputfile:
    for line in inputfile:
        lines.append(line.split(','))
        timestamp.append(float(lines[-1][1]))
        id.append(float(lines[-1][2]))
        w.append(float(lines[-1][3]))
        x.append(float(lines[-1][4]))
        y.append(float(lines[-1][5]))
        z.append(float(lines[-1][6]))
        acc_x.append(float(lines[-1][7]))
        acc_y.append(float(lines[-1][8]))
        acc_z.append(float(lines[-1][9]))

def simulation():
    global timestamp, id, w, x, y, z, acc_x, acc_y, acc_z, server, connection
    print('Sending data...')
    for i in range(1, len(timestamp)):
        msg = [timestamp[i], id[i], w[i], x[i], y[i], z[i], acc_x[i], acc_y[i], acc_z[i]]
        sleep(timestamp[i] - timestamp[i - 1])
        if connection:
            server.send(msg)



# Connect to the server
connection = False
try:
    address = ('localhost', 50001)
    # server = socket.socket()
    # server.connect(address)
    server = Client(address)
    server.send('imus')
    connection = True
except Exception as e:
    print('No server found in address {}'.format(address))

simulation()
