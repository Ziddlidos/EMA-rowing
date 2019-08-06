import cv2
import numpy
import matplotlib.pyplot as plt
import time
import signal
import sys
import cv2
import numpy
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    # find the webcam

    # video recorder
    capture = cv2.VideoCapture(2)

    # video recorder
    fourcc = cv2.cv.CV_FOURCC(*'X264')  # cv2.VideoWriter_fourcc() does not existit
    video_writer = cv2.VideoWriter("video.mp4", fourcc, 30, (680, 480))

    with open("video.time", 'w') as f:
        f.write("Video teste")

    # record video
    while (capture.isOpened()):
        start = time.time()
        ret, frame = capture.read()
        if ret:
            video_writer.write(frame)
            with open("video.time", '') as f:
                f.write(time.time())
        else:
            break

    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()