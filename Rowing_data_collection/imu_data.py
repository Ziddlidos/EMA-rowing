# this is a client for the IMUs

import time
import serial
import numpy as np
from multiprocessing.connection import Client
# import socket
import serial.tools.list_ports

#TODO stop IMUs and close connection to serial port and server on exit

connection = False
try:
    address = ('localhost', 5000)
    # server = socket.socket()
    # server.connect(address)
    server = Client(('localhost', 5000))
    server.send('imus')
except:
    print('No server found in address {}'.format(address))

addresses = [1,2]
command = 0

a = serial.tools.list_ports.comports()
for w in a:
    print("\tPort:", w.device, "\tSerial#:", w.serial_number, "\tDesc:", w.description, 'PID', w.pid)
    if w.pid == 4128: # small IMU dongle
        portIMU = w.device

# portIMU = '/dev/tty.usbmodem14101' # rPi
serial_port = serial.Serial(port=portIMU, baudrate=115200, timeout=0.001)
time.sleep(0.1)
serial_port.flush()
time.sleep(0.1)

# Set streaming slots
for i in range(len(addresses)):
    msg = '>'+str(addresses[i])+',80,'+str(command)+',255,255,255,255,255,255,255\n'
    print(msg)
    serial_port.write(msg.encode())
    time.sleep(0.1)
    out = ''
    while serial_port.inWaiting():
        out += '>> ' + serial_port.read(serial_port.inWaiting()).decode()
    print(out)
out = ''

# Start streaming
for i in range(len(addresses)):
    serial_port.write(('>'+str(addresses[i])+',85\n').encode())
    time.sleep(0.1)
    while serial_port.inWaiting():
        out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()

print('Start')

def read_sensors(portIMU):
    global x,y,z, running, counters
    t0 = time.time()
    running = True
    id = 0
    now = time.time()
    while running:

        bytes_to_read = serial_port.inWaiting()

        if bytes_to_read > 0:
            data = serial_port.read(bytes_to_read)
            data2 = data.decode().replace('\r\n',' ')
            # data2 = ''.join(chr(i) for i in data.encode() if ord(chr(i)) > 31 and ord(chr(i)) < 128 )
            data3 = data2.split(' ')
            data3 = list(filter(None, data3))
            # print(data3)

            temp = data3[-1]  # Get latest message and ignore others
            # print(temp)
            temp = temp[3:]  # Remove undesired first 3 bytes
            temp = temp.split(',')
            temp = np.array(temp).astype(np.float)
            x = temp[0]
            y = temp[1]
            z = temp[2]
            w = temp[3]

            id = data[1]
            out = [time.time(), id, x, y, z, w]
            if connection:
                server.send(out)
            # print(out)
            # print(1/(time.time()-now))
            now = time.time()

        else:
            # print("No data")
            # time.sleep(0.1)
            pass



read_sensors(portIMU)