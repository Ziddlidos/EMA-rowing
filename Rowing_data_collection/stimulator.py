# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:13:47 2015

@author: Wall-e
"""

# Stimulator functions

import time
import sys
import glob
import serial

class Stimulator:
    def __init__(self, port):
        self.serial_port = port
        

########################################
# Initialization
########################################


    def initialization(self, freq, channels):
    #    try:
        delay = 0.001
        #valores para teste#
        ts1 = round((1/float(freq))*1000)
    #    print ts1
        ts2 = 1.5
        main_time = int(round((ts1 - 1) / 0.5))
    #    main_time = int(round(ts1))
        group_time = int(round((ts2 - 1.5) / 0.5))
        channel_stim = channels
        channel_lf = 0
        n_factor = 0
        check = int((n_factor + channel_stim + channel_lf + group_time + main_time)%8)
        
        init_1 = (1<<7) | (0<<6) | (check<<2) | (n_factor>>1)
        init_2 = ((n_factor&1)<<6) | (channel_stim>>2)
        init_3 = ((channel_stim & 3) << 5) | (channel_lf >> 3)
        init_4 = ((channel_lf & 7) << 4) | (group_time >> 3)
        init_5 =  ((group_time & 7) << 4) | (main_time >> 7)
        init_6 = main_time & 127
    #        
    #        print(group_time)
    #        print(init_1)
    #        print(init_2)
    #        print(init_3)
    #        print(init_4)
    #        print(init_5)
    #        print(init_6)
        
        init = bytearray([init_1, init_2, init_3, init_4, init_5, init_6])
        
        dados = ""
        i=1
        self.serial_port.write(init) # e escreve na porta
        
        time.sleep(delay)
        while dados == "":            
            self.serial_port.flush()
            data = self.serial_port.read(self.serial_port.inWaiting()) # le da porta bytearray
            dados = data.decode()  # transforma bytearray em string
            i += 1
            if i > 700:
#                        print "Sem resposta."
                dados = 'No answer'
                return dados
        dados = data.decode()  # transforma bytearray em string
    #        print "resposta: " + dados
    #        print data
            
    #    except ValueError:
    #        print("Error (initialization)")
            
        return dados
        
        
        
########################################
# Update
########################################
        
    def update(self, channels, width, current):
    
        try:
    #        print "beggining update"
            #valores para teste#
    #        for i in width:
    #            pulse_width(i) = width
            pulse_width = width
            pulse_current = current
            mode = 0
    #        a = sum(pulse_width)
    #        print type(width[0])
    #        pulse_width = [200]
    #        pulse_current = [4]
            check = int((mode + sum(pulse_width) + sum(pulse_current))%32)
            init_b = []
            for i in range(len(width)*3+1):
                init_b.append(0)
            init_b[0] = (1<<7) | (1<<5) | (check) 
            for i in range(len(width)):         
                init_b[i*3+1] = (mode<<5) | (int(pulse_width[i])>>7)
                init_b[i*3+2] = (int(pulse_width[i]) & 127)
                init_b[i*3+3] = pulse_current[i]
            
    #        print(init_1)
    #        print(init_2)
    #        print(init_3)
    #        print(init_4)
            
    #        init(1) = init_b(1)
    
            init = bytearray(init_b)
            
            i=0
            
            
    #        print "requesting update"
            self.serial_port.flush()
            self.serial_port.write(init) # e escreve na porta
            dados = ""
    #        print "escreveu"
            while dados == "":            
                self.serial_port.flush()
                data = self.serial_port.read(self.serial_port.inWaiting()) # le da porta bytearray
                dados = data.decode()  # transforma bytearray em string
                i += 1
                if i > 700:
    #                        print "Sem resposta."
                    dados = 'No answer'
                    return dados       
                
            dados = data.decode()  # transforma bytearray em string


                  
        except ValueError:
            return 'Error'
            
        return dados
        
        
########################################
# Stop
########################################
           
    def stop(self):
        try:
            stop_1 = 192
            stop = bytearray([stop_1])  
            i=1
            self.serial_port.write(stop) # e escreve na porta
            data = ""
            dados = ""
            
            while dados == "":            
                self.serial_port.flush()
                data = self.serial_port.read(self.serial_port.inWaiting()) # le da porta bytearray
                dados = data.decode()  # transforma bytearray em string
                i += 1
                if i > 700:
    #                        print "Sem resposta."
                    dados = 'No answer'
                    return dados        
                
    #        dados = data.decode()  # transforma bytearray em string
    #        print "resposta: " + dados
            
        except ValueError:
            return 0
            
        return dados

def get_port():
    """Get the serial port where the device is connected. Only available on Windows and OSX
    :raises EnvironmentError:
        On unsupported or unknown platforms
    :return:
        The serial port where the device is connected
    """
    port = 0
    if sys.platform.startswith('darwin'):
        port = glob.glob('/dev/tty.usbserial-HMQYVD6B')[0]
    elif sys.platform.startswith('win'):
        # path = 'HARDWARE\\DEVICEMAP\\SERIALCOMM'
        # try:
        #     key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
        # except winreg.WindowsError:
        #     raise winreg.IterationError
        #
        # for i in itertools.count():
        #     try:
        #         val = winreg.EnumValue(key, i)
        #         yield (str(val[1]), str(val[0]))
        #     except EnvironmentError:
        #         break
        ports = ['COM%s' % (i + 1) for i in range(32)]
        for p in ports:
            try:
                s = serial.Serial(p)
                s.close()
                port = p
            except (OSError, serial.SerialException):
                pass
    return port