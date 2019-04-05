'''
This script is used to communicate with the rowing handle bar.
If connection == True, it sends the button pressing data to the server for logging.
Authors: Breno Ferreira and Lucas Fonseca
Contact: lucasfonseca@lara.unb.br
Date: Feb 25th 2019
'''

import stimulator
import serial
import time
# import bluetooth
import serial.tools.list_ports
import io
from multiprocessing.connection import Client
from multiprocessing.connection import Listener
import threading

# import socket

# TODO close connection to serial port on exit() and stop stimulation

stimulation = True
connection = True

controller_address = ('localhost', 50002)

running_flag = True

try:
    address = ('localhost', 50001)
    server = Client(address)

    # server = socket.socket()
    # server.connect(address)
    server.send('stim')
    connection = True

    # print(a)
except:
    print('No server found in address {}'.format(address))

a = serial.tools.list_ports.comports()
for w in a:
    print("\tPort:", w.device, "\tSerial#:", w.serial_number, "\tDesc:", w.description)
    if w.description == 'USB2.0-Serial':
        bd_addr = w.device
    elif w.description == 'USB <-> Stimu_Control':
        stimulatorPort = w.device

# bd_addr = '/dev/cu.usbserial-1410'
# stimulatorPort = 'stimPort'
# sock = serial.Serial(bd_addr, baudrate=9600, timeout=0.1)
# time.sleep(5)
current_str = [0, 0, 0, 0, 0, 0, 0, 0]
running = True

print("Conectando")

statSend = True
statWait = True

# sock.write(b'a')  # envia 'a' sinalizando a conexao para o controlador
# while statSend == True:
# time.sleep(1)
# TODO make handshake
'''
temp= sock.readline()
Temp = temp.decode()
Temp = temp[0:8]
if temp == 'conectou':
    statWait = False
    statSend = False     
'''
print("Conectado")

parametros = 'No parameters'
# statWait = True
# while statWait:
#     p = sock.readline()
#     parametros = p[0:28]
#     if len(parametros) > 1:
#         statWait = False

if stimulation:
    serialStimulator = serial.Serial(stimulatorPort, baudrate=115200, timeout=0.1)
    stim = stimulator.Stimulator(serialStimulator)  # chama a classe
# time.sleep(3)

print('recebeu parametros:')
print(parametros)

# flag = parametros
# Waits connection from the controller
print('Waiting for connection from controller')
serv = Listener(controller_address)
controller = serv.accept()
print('Connected to the controller')

def stim_setup():
    # print(flag)
    print('Hard coded parameters')
    current_CH12 = 0
    current_CH34 = 0
    current_CH56 = 0
    current_CH78 = 0
    pw = 400
    freq = 40
    mode = 3
    current_CH12 = 15 # int(flag[1:4])
    current_CH34 = 10 # int(flag[5:8])
    # current_CH56 = int(flag[9:12])
    # current_CH78 = int(flag[13:16])
    # pw = int(flag[17:20])
    # freq = int(flag[21:24])
    # mode = int(flag[25:28])
    print(current_CH12, current_CH34, pw, mode, freq)
    canais = channels(mode)

    if stimulation:
        # Os parametros sao frequencias e canais
        stim.initialization(freq, canais)

    return [current_CH12, current_CH34, current_CH56, current_CH78, pw, mode, canais]


# mode eh a quantidade de canais utilizados e channels e como a funcao stim.inicialization interpreta esse canais
# logo, eh necessario codificar a quantidade de canais nessa forma binaria ,o mais a esquerda eh o 8 e o mais a direita eh o 1

def channels(mode):
    this_channels = 0b11111111
    '''
    if mode == 1:
        channels = 0b00000011
    elif mode == 2:
        channels = 0b00001100
    elif mode == 3:
        channels = 0b00001111
    elif mode == 6:
        channels = 0b00111111
    elif mode == 8:
        channels = 0b11111111
    '''

    return this_channels


def change_current():
    global current_str, running_flag
    print('Thread started')
    while running_flag:
        current_str = input('New current: ')
        current_str = current_str.split()
        current_str = [int(x) for x in current_str]
        print(current_str)


# channels = 0b11111111

def running(current_CH12, current_CH34, current_CH56, current_CH78, pw, mode, this_channels):
    global current_str
    # cria um vetor com as correntes para ser usado pela funcao update
    current_str = [current_CH12, current_CH12, current_CH34, current_CH34, current_CH56, current_CH56, current_CH78,
                   current_CH78]

    '''
    if mode == 1:
        current_str.append(current_CH12)
        current_str.append(current_CH12)
    elif mode == 2:
        #current_str.append(current_CH12)
        #current_str.append(current_CH12)
        current_str.append(current_CH34)
        current_str.append(current_CH34)
    elif mode == 3: # Canais 1 e 2 terao corrente A e canais 3 e 4 corrent B
        current_str.append(current_CH12)
        current_str.append(current_CH12)
        current_str.append(current_CH34)
        current_str.append(current_CH34)
    '''
    # sock.write(b'a')  # envia 'a' sinalizando a conexao para o controlador
    # print("running")

    try:
        state = 0
        # print(state)
        stim_state = 'stop'
        pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
        starting_time = time.time()
        while state != 3:
            msg = controller.recv()
            if int(msg) == state:
                continue
            state = int(msg)
            # conversion
            if state == -1:
                state = 2
            # print(state)
            if mode == 1:  # Extensão B00000011
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [pw, pw, 0, 0, 0, 0, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
            elif mode == 2:  # Flexão B00001100
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, pw, pw, 0, 0, 0, 0]
            elif mode == 3:  # Extensão + Flexão B00001111
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [pw, pw, 0, 0, 0, 0, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, pw, pw, 0, 0, 0, 0]
            elif mode == 4:  # (Extensão & Aux_Ext) B00110011
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [pw, pw, 0, 0, pw, pw, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
            elif mode == 5:  # (Extensão & Aux_Ext) + Flexão B00111111
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [pw, pw, 0, 0, pw, pw, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, pw, pw, 0, 0, 0, 0]
            elif mode == 6:  # (Flexão & Aux_Flex) B11001100
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, pw, pw, 0, 0, pw, pw]
            elif mode == 7:  # Extensao + (Flexão & Aux_Flex) B11001111
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [pw, pw, 0, 0, 0, 0, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, pw, pw, 0, 0, pw, pw]
            elif mode == 8:  # (Extensão & Aux_Ext) + (Flexão & Aux_Flex) B11111111
                if state == 0:
                    stim_state = 'stop'
                    pw_str = [0, 0, 0, 0, 0, 0, 0, 0]
                # stim.stop()
                elif state == 1:
                    stim_state = 'extension'
                    pw_str = [pw, pw, 0, 0, pw, pw, 0, 0]
                elif state == 2:
                    stim_state = 'flexion'
                    pw_str = [0, 0, pw, pw, 0, 0, pw, pw]
                # para usar 6 ou 8 canais eh necessario copiar o codigo logo acima e mudar somente o vetor pw,
                # colocando-se pw no canal que se quer estimular
            if stimulation:
                stim.update(this_channels, pw_str, current_str)
            print(stim_state, current_str)
            if connection:
                # server.send(dict({'state':'Flexão', 'current':current_str}))
                server.send([time.time(), stim_state, current_str])
    except:
        print('Lost connection to controller')


def main():
    global running_flag
    [current_CH12, current_CH34, current_CH56, current_CH78, pw, mode, channels] = stim_setup()
    print(current_CH12, current_CH34, current_CH56, current_CH78, pw, mode, channels)

    t = threading.Thread(target=change_current)
    t.start()
    running(current_CH12, current_CH34, current_CH56, current_CH78, pw, mode, channels)

    if stimulation:
        stim.stop()
        serialStimulator.close()
    # sock.close()

    running_flag = False
    print("End")


if __name__ == '__main__':
    main()
