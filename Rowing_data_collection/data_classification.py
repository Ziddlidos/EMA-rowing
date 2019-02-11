class Cdata:
    def __init__(self, classification):
        self.classification = classification
        self.timestamp = []
        self.values = []


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
                while buttons_timestamp[position-1] == buttons_timestamp[position]:
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
