'''
This script loads rowing data that was saved to a file and saves it to a new file in binary form.
This makes it much faster to load and further process later.
Author: Lucas Fonseca
Contact: lucasfonseca@lara.unb.br
Date: Feb 25th 2019
'''

from data_processing import *
from PyQt5.QtWidgets import QApplication
import pickle
import sys


if __name__ == '__main__':
    app = QApplication(sys.argv)
    source_files = GetFilesToLoad()

    [emg_files, imus_files, buttons_files] = separate_files(source_files.filename[0])

    starting_time = get_starting_time([buttons_files[0], imus_files[0]])

    [buttons_timestamp, buttons_values] = parse_button_file(buttons_files[0], starting_time)
    imus = parse_imus_file(imus_files[0], starting_time)
    # [emg_1_timestamp, emg_1_values] = parse_emg_file(emg_files[0], starting_time)
    # [emg_2_timestamp, emg_2_values] = parse_emg_file(emg_files[1], starting_time)

    target_file = GetFileToSave()

    with open(target_file.filename[0], 'wb') as f:
        pickle.dump('buttons_timestamp', f)
        pickle.dump(buttons_timestamp, f)
        pickle.dump('buttons_values', f)
        pickle.dump(buttons_values, f)
        pickle.dump('imus', f)
        pickle.dump(imus, f)
        pickle.dump('emg_1_timestamp', f)
        # pickle.dump(emg_1_timestamp, f)
        # pickle.dump('emg_1_values', f)
        # pickle.dump(emg_1_values, f)
        # pickle.dump('emg_2_timestamp', f)
        # pickle.dump(emg_2_timestamp, f)
        # pickle.dump('emg_2_values', f)
        # pickle.dump(emg_2_values, f)
