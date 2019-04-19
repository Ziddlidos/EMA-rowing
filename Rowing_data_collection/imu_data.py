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
command = [255, 255, 255, 255, 255, 255, 255, 255]
command[0] = 0
command[1] = 41
command[2] = 255
command[3] = 255
command[4] = 255
command[5] = 255
command[6] = 255
command[7] = 255


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
    msg = '>' + str(addresses[i]) + ',80,' + str(command[0]) + ',' + \
                                             str(command[1]) + ',' + \
                                             str(command[2]) + ',' + \
                                             str(command[3]) + ',' + \
                                             str(command[4]) + ',' + \
                                             str(command[5]) + ',' + \
                                             str(command[6]) + ',' + \
                                             str(command[7]) + '\n'
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
# for i in range(len(addresses)):
#     serial_port.write(('>'+str(addresses[i])+',165\n').encode())
#     time.sleep(0.1)
#     while serial_port.inWaiting():
#         out = '>> ' + serial_port.read(serial_port.inWaiting()).decode()

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
                # print('Full raw data: ' + str(data))
                # print(data[0])
                if len(data) <= 3 or data[0] != 0:
                    continue
                data2 = data.decode().replace('\r\n',' ')
                # data2 = ''.join(chr(i) for i in data.encode() if ord(chr(i)) > 31 and ord(chr(i)) < 128 )
                data3 = data2.split(' ')
                data3 = list(filter(None, data3))
                # print(data3)

                info = data3[0][0:3]

                id = int.from_bytes(info[1].encode(), sys.byteorder)

                ################################################################
                ################################################################
                # get data
                quaternion = data3[0][3:]
                accel = data3[1]
                # id = int.from_bytes(temp[1].encode(), sys.byteorder)
                # print('Final data: ' + str(temp))
                # print('Quaternion: ' + str(quaternion))
                # print('Accel: ' + str(accel))
                # print(temp[1].encode())
                # print(id)
                # print(quaternion)

                quaternion = quaternion.split(',')
                quaternion = np.array(quaternion).astype(np.float)

                accel = accel.split(',')
                accel = np.array(accel).astype(np.float)


                x = quaternion[0]
                y = quaternion[1]
                z = quaternion[2]
                w = quaternion[3]
                acc_x = accel[0]
                acc_y = accel[1]
                acc_z = accel[2]

                ################################################################
                ################################################################


                # Send data to server
                out = [time.time(), id, w, x, y, z, acc_x, acc_y, acc_z]
                if connection:
                    server.send(out)
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