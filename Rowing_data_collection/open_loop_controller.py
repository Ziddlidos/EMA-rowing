import time
from multiprocessing.connection import Client
import os

stroke_duration = 7
drive_duration = 3
extension_duration = 4
flexion_duration = stroke_duration - extension_duration
total_duration = 30

connection = True

a_sound = 'a_sound.mp3'
b_sound = 'b_sound.mp3'

try:
    address = ('localhost', 50002)
    server = Client(address)
    # server.send('stim')
    # connection = True
    print('Connected to stim')

    # print(a)
except:
    print('No server found in address {}'.format(address))
    quit()

def play_sound(sound):
    os.system('mpg123 ' + sound)

print('Press ENTER to start')
input()
now = time.time()
starting_time = now
current_state = 1
server.send(current_state)
play_sound(a_sound)
while True:
    if current_state == 1:
        while time.time() - now < extension_duration:
            pass
        current_state = -1
        server.send(current_state)
        now = time.time()
        play_sound(b_sound)
    elif current_state == -1:
        while time.time() - now < flexion_duration:
            pass
        current_state = 1
        server.send(current_state)
        now = time.time()
        play_sound(a_sound)
    if time.time() - starting_time > total_duration:
        break

print('End of practice')
print('Good job!')
