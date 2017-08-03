"""
Author: Pedro Paulo Sanches - EMA - LARA - Uiversidade de Brasília
Editado em 05/06/2017


"""
import stimulator
import serial
import time
import bluetooth
# python -m serial.tools.list_ports    para saber as portas no Linux Ou py no Windows

arduinoPort = 'COM3' #'/dev/ttyACM0' linux
stimulatorPort = 'COM4' #'/dev/ttyUSB0' linux

"""
canais:
1 e 2 = quadriceps
3 e 4 = isquiotibiais
"""
running = True
stimulation = True
pw = 400 #pulse width
current = 30
current_str = [current,current]
buttonState = 0
freq = 50
channels = 0b00000011 #Forma binaria de escolher os canais utilizados (o mais a esquerda é o 8 e o mais a direita é o 1)

def modules_setup(channels,arduinoPort,stimulatorPort):

    #abre porta serial para o arduino
    serialArduino = serial.Serial(arduinoPort, baudrate = 9600 , timeout = 1)
    
    #abre porta serial para o estimulador
    serialStimulator = serial.Serial(stimulatorPort, baudrate=115200, timeout=1)
    stim = stimulator.Stimulator(serialStimulator) #chama a classe
    
    # Os parametros sao frequencias e canais, nao sei se e a quantidade de canais ou o canal especifico
    stim.initialization(freq,channels)


def read_controlers():
    buttonState = serialArduino.readline()
    return buttonState

def main():
    modules_setup(channels,arduinoPort,stimulatorPort)
    while 1:

        try:
            state = int(read_controlers())
        except ValueError:  
            state = 0
            pass

        if state == 0: 
            print("Parado")
            stim.update(channels,[0,0], current_str)
           # stim.stop()
        elif state == 1:
            stim.update(channels, [pw,pw], current_str)    
            print("Estimulo em Quadriceps")       
        elif state == 2:
            stim.update(channels, [0,0], current_str)    
         #   print("Estimulo em Isquiotibiais")    
        elif state == 3:
            stim.stop()
            print("Saiu")
            serialArduino.close()
            serialStimulator.close()
            break




if __name__ == '__main__':
    main()

