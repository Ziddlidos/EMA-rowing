'''
This script is used to comunicate with the IMUs.
If connection == True, it sends data to the server for logging.
Authors: Lucas Fonseca
Contact: lucasfonseca@lara.unb.br
Date: Feb 25th 2019
Last update: April 12th 2019
'''
# this is a client for the IMUs

import time
import serial
import numpy as np
from multiprocessing.connection import Client
import serial.tools.list_ports
import sys

#TODO stop IMUs and close connection to serial port and server on exit

# Connect to the server
connection = False
try:
    address = ('localhost', 50001)
    # server = socket.socket()
    # server.connect(address)
    server = Client(address)
    server.send('imus')
    connection = True
except:
    print('No server found in address {}'.format(address))

addresses = [1,2,3,4,5,6,7,8]

# Command the IMUs are to perform
command = 0

# Find and open serial port for the IMU dongle
a = serial.tools.list_ports.comports()
for w in a:
    print("\tPort:", w.device, "\tSerial#:", w.serial_number, "\tDesc:", w.description, 'PID', w.pid)
    if w.pid == 4128: # small IMU dongle
        portIMU = w.device
# portIMU = '/dev/tty.usbmodem14101' # rPi
serial_port = serial.Serial(port=portIMU, baudrate=115200, timeout=0.01)
time.sleep(0.1)
serial_port.flush()
serial_port.flushInput()
serial_port.flushOutput()
time.sleep(0.1)


# Stop streaming
for i in range(len(addresses)):
    serial_port.write(('>'+str(addresses[i])+',86\n').encode())
    time.sleep(0.1)
    while serial_port.inWaiting():
        out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()
    # print(out)

# Manual flush. Might not be necessary
while not serial_port.inWaiting() == 0:
    serial_port.read(serial_port.inWaiting())
    time.sleep(0.1)

print('Starting configuration')
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

# Set mag on(1)/off(0)
for i in range(len(addresses)):
    serial_port.write(('>'+str(addresses[i])+',109, 0\n').encode())
    time.sleep(0.1)
    while serial_port.inWaiting():
        out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()

# Gyro autocalibration
for i in range(len(addresses)):
    serial_port.write(('>'+str(addresses[i])+',165\n').encode())
    time.sleep(0.1)
    while serial_port.inWaiting():
        out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()

# Tare
for i in range(len(addresses)):
    serial_port.write(('>'+str(addresses[i])+',96\n').encode())
    time.sleep(0.1)
    while serial_port.inWaiting():
        out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()
    # print(out)

# Start streaming
for i in range(len(addresses)):
    serial_port.write(('>'+str(addresses[i])+',85\n').encode())
    time.sleep(0.1)
    while serial_port.inWaiting():
        out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()
    # print(out)

print('Start')

def read_sensors(portIMU):
    global x,y,z, running, counters
    t0 = time.time()
    running = True
    id = 0
    now = time.time()
    try:
        while running:
            # print('waiting...')
            bytes_to_read = serial_port.inWaiting()
            # print(bytes_to_read)
            if bytes_to_read > 0:
                # print('reading...')
                data = serial_port.read(bytes_to_read)

                data2 = data.decode().replace('\r\n',' ')
                # data2 = ''.join(chr(i) for i in data.encode() if ord(chr(i)) > 31 and ord(chr(i)) < 128 )
                data3 = data2.split(' ')
                data3 = list(filter(None, data3))
                # print(data3)

                temp = data3[-1]  # Get latest message and ignore others
                # print(temp[1].encode())
                id = int.from_bytes(temp[1].encode(), sys.byteorder)
                # print(id)
                temp = temp[3:]  # Remove undesired first 3 bytes
                temp = temp.split(',')
                temp = np.array(temp).astype(np.float)
                if len(temp) == 4:
                    x = temp[0]
                    y = temp[1]
                    z = temp[2]
                    w = temp[3]
                else:
                    continue

                # id = data[1]
                out = [time.time(), id, w, x, y, z]
                if connection:
                    server.send(out)
                # print(out)
                # print(1/(time.time()-now))
                now = time.time()

            else:
                # print("No data")
                # time.sleep(0.1)
                pass


    except Exception as e:
        print('Exception raised: ', str(e), ', on line ', str(sys.exc_info()[2].tb_lineno))
        # Stop streaming
        for i in range(len(addresses)):
            serial_port.write(('>' + str(addresses[i]) + ',86\n').encode())
            time.sleep(0.1)
            while serial_port.inWaiting():
                out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()
            # print(out)
        serial_port.close()

read_sensors(portIMU)