# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np


# Load yolo5 model and find result from nn
def score_frame(frame):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = [frame]
    return model(frame)

# Modify frame
def clipping_frame(
    results, frame, x_shape, y_shape, id_ob):
    # Conteiner for frames
    frames = []
    for i in results.xyxy[0]:
        # If detected object from class what we have
        if int(i[5]) == id_ob:
            x1, y1, x2, y2 = int(i[0]), int(i[1]),int(i[2]), int(i[3])

            frame_copy = frame
            
            frame_copy[:y1][:] = 0
            frame_copy[y2:][:] = 0

            for p in range(y1, y2):
                for k in range (0, x_shape):
                    if not (k >= x1 and k <= x2 and p >= y1 and p<= y2):
                        frame_copy[p][k] = 0

            frames.append(frame_copy) 
    
    result = np.zeros_like(frame)
    for i in frames:
        result = cv2.bitwise_or(result, i)
    return result


if __name__ == '__main__':
    def nothing(*arg):
        pass

    print('Enter path to video:')
    path = input() #'Roads/1.mp4'

    # Setting up the camera
    cap = cv2.VideoCapture(path)

    # Enter class of searching object
    print('Enter type of searching object: 0 - bird, 1 - cat, 2 - dog')
    search_ind = int(input()) + 14

    if (search_ind > 16 or search_ind < 14):
        print('Can`t do it')
    else:

        # Find size of input video
        x_shape, y_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20, (x_shape, y_shape))

        while cap.isOpened():

            # Getting an image from the file
            flag, frame = cap.read()
            if not flag:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Get result from nn    
            results = score_frame(frame)

            # Get result frame after processing
            frame = clipping_frame(results, frame, x_shape, y_shape, search_ind)

            
            cv2.imshow('IMG', frame)

            # Write result frame in file
            out.write(frame)
            ch = cv2.waitKey(1)
            # to exit, press esc
            if ch == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()