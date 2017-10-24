"""
Author: Pedro Paulo Sanches - EMA - LARA - Uiversidade de Brasilia
Editado em 05/06/2017


"""
import stimulator
import serial
import time
import bluetooth
# python -m serial.tools.list_ports    para saber as portas no Linux Ou py no Windows

stimulatorPort = '/dev/ttyUSB0' #linux
bd_addr = '98:D3:32:10:B0:63'#endereco do HC-05 conectado ao arduino
port = 1

running = True
stimulation = True

#inicializa conexao bluetooth
sock = bluetooth.BluetoothSocket (bluetooth.RFCOMM)
sock.connect((bd_addr,port))

print("Conectando")
sock.send("a") # envia 'a' sinalizando a conexao para o controlador
sock.recv(1)	
print("Conectado")

#abre porta serial para o estimulador
serialStimulator = serial.Serial(stimulatorPort, baudrate=115200, timeout=1)
stim = stimulator.Stimulator(serialStimulator) #chama a classe

    
def stim_setup():
    
    for cont in range(4):
        flag = sock.recv(1)
        print(flag)
        if flag == b'c':
            current = int(sock.recv(3))
            time.sleep(0.5)
        elif flag == b'p':
            pw = int(sock.recv(3))
            time.sleep(0.5)
        elif flag == b'f':
            freq = int(sock.recv(3))
            time.sleep(0.5)
        elif flag == b'm':
            mode = int(sock.recv(3))
            time.sleep(0.5)


    print(current,pw,mode,freq)
    canais = channels(mode)
    
    # Os parametros sao frequencias e canais
    stim.initialization(freq,canais)

    return [current,pw,mode,canais]

# mode eh a quantidade de canais utilizados e channels e como a funcao stim.inicialization interpreta esse canais
# logo, eh necessario codificar a quantidade de canais nessa forma binaria ,o mais a esquerda eh o 8 e o mais a direita eh o 1
def channels(mode):
    if mode == 2:
        channels = 0b00000011
    elif mode == 4:
        channels = 0b00001111
    elif mode == 6:
        channels = 0b00111111
    elif mode == 8:
        channels = 0b11111111

    return channels

def running(current,pw,mode,channels):
    
    #cria um vetor com as correntes para ser usado pela funcao update
    current_str = []
    for n in range(mode):
        current_str.append(current)
        
    sock.send("a") # envia 'a' sinalizando o inicio da estimulacao
    print("running")
    state = int(sock.recv(1))
    print(state)
    while state != 3:
        state = int(sock.recv(1))
        print(state)
        if mode == 2:                           # Para 2 canais
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels, [pw,pw], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels, [0,0], current_str)    
                print("Contracao")    
        elif mode == 4:                         # Para 4 canais
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[0,0,pw,pw], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[pw,pw,0,0], current_str)    
                print("Contracao")    
            #para usar 6 ou 8 canais eh necessario copiar o codigo logo acima e mudar somente o vetor pw,
            #colocando-se pw no canal que se quer estimular
    
def main():
    [current,pw,mode,channels] = stim_setup()
    print(current,pw,mode,channels)
    running(current,pw,mode,channels)

    stim.stop()
    sock.close()
    serialStimulator.close()
    print("Saiu")

if __name__ == '__main__':
    main()

