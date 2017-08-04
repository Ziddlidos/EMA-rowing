import bluetooth
import time

bd_addr = '98:D3:32:20:A2:5B'
port = 1
sock = bluetooth.BluetoothSocket (bluetooth.RFCOMM)
sock.connect((bd_addr,port))

current = 0
pw = 0
freq = 0
mode = 0


def conect():
    print("Conectando")
    sock.send("a") # envia 1 sinalizando a conexao para o controlador
    sock.recv(1)	
    print("Conectado")

# Pega os valores de corrente, largura de pulso, frequencia e modo de operacao do sistema
def setup():
    for cont in range(4):
        flag = sock.recv(1)
        if flag == 'c':
            print(flag)
            current = sock.recv(3)
            print(current)
            time.sleep(0.1)
        elif flag == 'p':
            print(flag)
            pw = sock.recv(3)
            print(pw)
            time.sleep(0.1)
        elif flag == 'f':
            print(flag)
            freq = sock.recv(3)
            print(freq)
            time.sleep(0.1)
        elif flag == 'm':
            print(flag)
            mode = sock.recv(3)
            print(mode)
            time.sleep(0.1)
    return [current,pw,freq,mode]


def main():
    conect()
    [current,pw,freq,mode] = setup()
    print(current, pw, freq, mode)
    sock.close()

if __name__ == '__main__':
    main()
