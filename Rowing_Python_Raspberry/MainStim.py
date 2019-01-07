import stimulator
import serial
import time
#import bluetooth
import serial.tools.list_ports
import io

a=serial.tools.list_ports.comports()
#print(a)

for w in a:
    print("\tPort:", w.device,"\tSerial#:", w.serial_number, "\tDesc:", w.description)
    if w.description == 'USB2.0-Serial':
        bd_addr = w.device
    elif w.description == 'USB <-> Stimu_Control':
        stimulatorPort = w.device
   
sock = serial.Serial(bd_addr, baudrate=9600, timeout=0.1)
time.sleep(15)

running = True
stimulation = True

print("Conectando")

statSend = True
statWait = True

sock.write(b'a') # envia 'a' sinalizando a conexao para o controlador
#while statSend == True:
time.sleep(1)
'''
temp= sock.readline()
Temp = temp.decode()
Temp = temp[0:8]
if temp == 'conectou':
    statWait = False
    statSend = False     
'''
print("Conectado")

statWait = True
while statWait == True:
    parametros = sock.readline().decode()
    parametros = parametros[0:28]
    if parametros != '':
        statWait = False
        
serialStimulator = serial.Serial(stimulatorPort, baudrate=115200, timeout=0.1)
stim = stimulator.Stimulator(serialStimulator) #chama a classe
time.sleep(5)


print('recebeu parametros:')
print(parametros)

flag = parametros

   
def stim_setup():
    print(flag)
    current_CH12 = int(flag[1:4])
    current_CH34 = int(flag[5:8])
    current_CH56 = int(flag[9:12])
    current_CH78 = int(flag[13:16])
    pw = int(flag[17:20])
    freq = int(flag[21:24])
    mode = int(flag[25:28])
    print(current_CH12,current_CH34,pw,mode,freq)
    canais = channels(mode)
    
    # Os parametros sao frequencias e canais
    stim.initialization(freq,canais)

    return [current_CH12,current_CH34,current_CH56,current_CH78,pw,mode,canais]

# mode eh a quantidade de canais utilizados e channels e como a funcao stim.inicialization interpreta esse canais
# logo, eh necessario codificar a quantidade de canais nessa forma binaria ,o mais a esquerda eh o 8 e o mais a direita eh o 1

def channels(mode):

    channels = 0b11111111
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

    return channels

#channels = 0b11111111

def running(current_CH12,current_CH34,current_CH56,current_CH78,pw,mode,channels):
    
    #cria um vetor com as correntes para ser usado pela funcao update
    current_str = []
    current_str.append(current_CH12)
    current_str.append(current_CH12)
    current_str.append(current_CH34)
    current_str.append(current_CH34)
    current_str.append(current_CH56)
    current_str.append(current_CH56)
    current_str.append(current_CH78)
    current_str.append(current_CH78)


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
    sock.write(b'a') # envia 'a' sinalizando a conexao para o controlador
    print("running")
    
    
    state = 0
    print(state)
    while state != 3:
        while sock.inWaiting() == 0:
            pass
        state = int(sock.read(1))#state = int(sock.read(1))
        print(state)
        if mode == 1:       # Extensão B00000011 
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[pw,pw,0,0,0,0,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)    
                print("Flexao")    
        elif mode == 2:     # Flexão B00001100
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
                # stim.stop()
            elif state == 1:
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,pw,pw,0,0,0,0], current_str)    
                print("Flexao")
        elif mode == 3:     # Extensão + Flexão B00001111
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[pw,pw,0,0,0,0,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,pw,pw,0,0,0,0], current_str)    
                print("Flexao")
        elif mode == 4:     # (Extensão & Aux_Ext) B00110011
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[pw,pw,0,0,pw,pw,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)    
                print("Flexao")
        elif mode == 5:     # (Extensão & Aux_Ext) + Flexão B00111111
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[pw,pw,0,0,pw,pw,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,pw,pw,0,0,0,0], current_str)    
                print("Flexao")    
        elif mode == 6:     # (Flexão & Aux_Flex) B11001100
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,pw,pw,0,0,pw,pw], current_str)    
                print("Flexao")
        elif mode == 7:     # Extensao + (Flexão & Aux_Flex) B11001111
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[pw,pw,0,0,0,0,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,pw,pw,0,0,pw,pw], current_str)    
                print("Flexao")  
        elif mode == 8:     # (Extensão & Aux_Ext) + (Flexão & Aux_Flex) B11111111
            if state == 0: 
                print("Parado")
                stim.update(channels,[0,0,0,0,0,0,0,0], current_str)
               # stim.stop()
            elif state == 1:
                stim.update(channels,[pw,pw,0,0,pw,pw,0,0], current_str)    
                print("Extensao")       
            elif state == 2:
                stim.update(channels,[0,0,pw,pw,0,0,pw,pw], current_str)    
                print("Flexao")   
        #para usar 6 ou 8 canais eh necessario copiar o codigo logo acima e mudar somente o vetor pw,
            #colocando-se pw no canal que se quer estimular
    
def main():
    [current_CH12,current_CH34,current_CH56,current_CH78,pw,mode,channels] = stim_setup()
    print(current_CH12,current_CH34,current_CH56,current_CH78,pw,mode,channels)
    running(current_CH12,current_CH34,current_CH56,current_CH78,pw,mode,channels)

    stim.stop()
    sock.close()
    serialStimulator.close()
    print("Saiu")

if __name__ == '__main__':
    main()



    


    