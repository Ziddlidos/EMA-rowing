'''
This file contains a collection of methods ans a class to help classifying rowing data.
Author: Lucas Fonseca
Contact: lucasafonseca@lara.unb.br
Date: Feb 25th 2019
'''

class Cdata:
    def __init__(self, classification):
        self.classification = classification
        self.timestamp = []
        self.values = []

class Classifier:

    def __init__(self, lda):
        self.lda = lda

    # classify given values with all available LDAs and returns predicted classes and probabilities
    def classify(self, values):
        import numpy as np
        out_class = []
        out_probability = []
        try:
            for l in self.lda:
                out_class.append(l.predict(np.array(values).reshape(1, -1)))
                out_probability.append(max(max(l.predict_proba(np.array(values).reshape(1, -1)))))
        except Exception:
            print(values)
            raise Exception
        return [out_class, out_probability]

# Classify data according to which button was beeing pressed at that instant
# Returns 3 lists of Cdatas. One for each classification
# TODO: make this return any number of classes
def classify_by_buttons(buttons_timestamp, buttons_values, vector_timestamp, vector_values):
    last_position = len(buttons_timestamp) - 1
    position = 0
    classification = buttons_values[position]
    low = []
    zero = []
    up = []
    this_class = Cdata(classification)
    if classification == -1:
        low.append(this_class)
    elif classification == 0:
        zero.append(this_class)
    elif classification == 1:
        up.append(this_class)
    for i in range(len(vector_timestamp)):
        timestamp = vector_timestamp[i]
        if timestamp < buttons_timestamp[position]:
            this_class.timestamp.append(timestamp)
            this_class.values.append(vector_values[i])
        else:
            if position < last_position:
                position += 1
                # while buttons_timestamp[position-1] == buttons_timestamp[position]:
                #     position += 1
                while buttons_values[position-1] == buttons_values[position] and position < last_position:
                    position += 1
                classification = buttons_values[position]
                this_class = Cdata(classification)
                if classification == -1:
                    low.append(this_class)
                elif classification == 0:
                    zero.append(this_class)
                elif classification == 1:
                    up.append(this_class)
                if timestamp < buttons_timestamp[position]:
                    this_class.timestamp.append(timestamp)
                    this_class.values.append(vector_values[i])
            else:
                break
    return [low, zero, up]

# Classify data according to which button was beeing pressed at that instant
# Returns a single list with the classification. This list is the same size as the input
def classify_by_buttons_in_order(buttons_timestamp, buttons_values, vector_timestamp):
    from numpy import ones
    last_position = len(buttons_timestamp)
    position = 0
    classification = [buttons_values[position]]
    position += 1

    for i in range(len(vector_timestamp)):
        timestamp = vector_timestamp[i]
        if timestamp < buttons_timestamp[position]:
            classification.append(buttons_values[position-1])
        else:
            classification.append(buttons_values[position])
            position += 1
            if position == last_position:
                break
    total_size = len(vector_timestamp)
    current_size = len(classification)
    if current_size < total_size:
        tail = ones(total_size - current_size) * classification[-1]
        classification += list(tail)
    return classification

# Separate any data into 3 different lists according to pre calculated classification
def separate_by_classification(vector_timestamp, vector_values, vector_classification):
    vector_low_timestamp = []
    vector_low_values = []
    vector_zero_timestamp = []
    vector_zero_values = []
    vector_up_timestamp = []
    vector_up_values = []

    for i in range(len(vector_timestamp)):
        if vector_classification[i] == 1:
            vector_up_timestamp.append(vector_timestamp[i])
            vector_up_values.append(vector_values[i])
        elif vector_classification[i] == 0:
            vector_zero_timestamp.append(vector_timestamp[i])
            vector_zero_values.append(vector_values[i])
        elif vector_classification[i] == -1:
            vector_low_timestamp.append(vector_timestamp[i])
            vector_low_values.append(vector_values[i])

    return [vector_low_timestamp,
            vector_low_values,
            vector_zero_timestamp,
            vector_zero_values,
            vector_up_timestamp,
            vector_up_values]

