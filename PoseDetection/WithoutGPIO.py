import cv2
import mediapipe as mp
import numpy as np
import time
import os
# import RPi.GPIO as GPIO
import serial

#for ultrasonic sensor
# GPIO.setmode(GPIO.BCM)
 
#set GPIO Pins
# GPIO_TRIGGER = 18
# GPIO_ECHO = 24
 
#set GPIO direction (IN / OUT)
# GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
# GPIO.setup(GPIO_ECHO, GPIO.IN)

def distance():
    # set Trigger to HIGH
    # GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    # time.sleep(0.00001)
    # GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    # while GPIO.input(GPIO_ECHO) == 0:
    #     StartTime = time.time()
    #
    # # save time of arrival
    # while GPIO.input(GPIO_ECHO) == 1:
    #     StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
body_pose = mp_pose.Pose(min_detection_confidence=0.3,min_tracking_confidence=0.3)


mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'XVID')

global recording_flag
global txtisdone
txtisdone = False
recording_flag = False
right_flag = False
left_flag = False

def Record():
    global recording_flag
    global output
    global txtisdone
    #for gsm module
    # if txtisdone == False:
    #     port = serial.Serial("/dev/ttyUSB1", baudrate=9600, timeout=1)
    #     port.write(b'AT\r\n')
    #     time.sleep(0.5)
    #     port.write(b"AT+CMGF=1\r")
    #     time.sleep(0.5)
    #     port.write(b'AT+CMGS="+639951053257"\r')
    #     time.sleep(0.5)
    #     msg = "Intruder Detected. Capturing Feed From Camera"
    #     port.reset_output_buffer()
    #     time.sleep(0.5)
    #     port.write(str.encode(msg+chr(26)))
    #     print("DONE")
    #     txtisdone = True
    if recording_flag == False:
        # we are transitioning from not recording to recording
        print("start")
        isexist = True
        number = 1
        while (isexist):
            number += 1
            path = 'captures/' + str(number) + '.avi'
            isexist = os.path.exists(path)

        output = cv2.VideoWriter(path, codec, 20.0, (640, 480))
        recording_flag = True


while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result

    body_results = body_pose.process(image)
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    if recording_flag:
        output.write(image)
    if body_results.pose_landmarks:
        dist = distance()
        # if (dist < 5):
        #     Record()
        #print(dist)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #print(face_landmarks)
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        #print(face_landmarks,x,y)
                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                text="forward"
                # See where the user's head tilting
                if y < -15:
                    #print("left")
                    if right_flag == True:
                        Record()
                    else:
                        left_flag = True
                elif y > 15:
                    #print("right")
                    text = "Looking Right"
                    if left_flag == False:
                        right_flag = True
                    else:
                        Record()
                elif x < -10:
                    #print("down")
                    text = "Looking Down"
                elif x > 10:
                    #print("up")
                    text = "UP"
                else:
                    #print("forward")
                    text = "Forward"

                # Display the nose direction
                #nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                #p1 = (int(nose_2d[0]), int(nose_2d[1]))
                #p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                #cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                #cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                #cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #end = time.time()
            #totalTime = end - start

            #fps = 1 / totalTime
            # print("FPS: ", fps)

            #cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            #mp_drawing.draw_landmarks(
            #    image=image,
            #    landmark_list=face_landmarks,
             #   connections=mp_face_mesh.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=drawing_spec,
            #    connection_drawing_spec=drawing_spec)
    else:
        print("No Detection")
        recording_flag = False
        right_flag = False
        left_flag = False
        txtisdone = False
        

    cv2.imshow('Head Pose Estimation', image)


    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
