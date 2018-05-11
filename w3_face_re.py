import cv2
from subprocess import call
import os
import numpy as np
import time
import threading
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
TRIG = 19
ECHO = 26
pulse_end_01 =0.00
pulse_start_01 = 0.00
pulse_duration =0.00
alt_holt_code=456
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
sensor_enable=False

subjects = ["", "Borhan","Tawhid", "Arif_iOS"]
def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    global face_cascade
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]
def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    counter=0
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            print("....Processing....", counter)
            counter=counter+1
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def speech(str):
  cmd_beg = 'espeak -g5 -ven+f4 '
  cmd_end = ' 2>/dev/null'  # To dump the std errors to /dev/null
  str = str.replace(' ', '_')
  str = str.replace(' ', '.')
  str = str.replace(' ', ',')
  call([cmd_beg  + str + cmd_end], shell=True)

def predict(test_img, catch=None):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    print (img)
    face, rect = detect_face(img)
    print ("BB")
    # predict the image using our face recognizer
    try:
        label, confidence = face_recognizer.predict(face)
        # get name of respective label returned by face recognizer
        label_text = subjects[label]
        speech("hi " + label_text)
        print (label_text, " ", confidence)
        # draw a rectangle around face detected
        draw_rectangle(img, rect)
        # draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1] - 5)
        return img
    except ValueError:
        print (ValueError)
        return None

print("Predicting images...")

def read_sensor():
    fleft=False
    while True:
        print("borhan")
        if sensor_enable:
            # print ("dfsfdsf")
            GPIO.output(TRIG, False)
            time.sleep(0.1)
            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)
            GPIO.setwarnings(False)

            while GPIO.input(ECHO) == 0:
                global pulse_start_01
                pulse_start_01 = time.time()
                # print ("Start ", pulse_start)
            while GPIO.input(ECHO) == 1:
                global pulse_end_01
                pulse_end_01 = time.time()

            pulse_duration = pulse_end_01 - pulse_start_01
            distance = pulse_duration * 17150
            distance = round(distance, 2)

            if distance > 10 and distance < 400:
                if (distance < 100):
                    print ("de")
                    #global obsCounter
                    #obsCounter=obsCounter+1
                    #object_dectec()
                    #obs_test()
                #print ("Distance:", distance - 0.5, "cm", obsCounter," ", modeValue)
                print ("Distance:", distance - 0.5, "cm"," ")
            else:
                if (fleft):
                    fleft = False
thread = threading.Thread(target=read_sensor)
thread.daemon = True
thread.start()

# while(1):
#       keypressed = raw_input()
#       if keypressed == 'q':
#            break
#            pass
#       elif keypressed == 'a':
#           #os.system("raspistill -w 480 -h 640 -o cam.jpg")
#           #os.system("raspistill -w 480 -h 640 -o cam.jpg")
#           os.system("raspistill -w 480 -h 640 -o cam.jpg")
#           time.sleep(1)
#           test_img = cv2.imread("cam.jpg")
#           #test_img =cv2.imread("cam.jpg")
#           predicted_img = predict(test_img)
#           if predicted_img is None:
#               print ("I do not know you ")
#           else:
#               cv2.imshow(subjects[1], cv2.resize(predicted_img, (480, 640)))
#               cv2.waitKey(0)
#               # cv2.destroyAllWindows()
#               # cv2.waitKey(1)
#               # cv2.destroyAllWindows()
#       elif keypressed == 'b':
#           test_img =cv2.imread("test-data/tawhid.png")
#           predicted_img = predict(test_img)
#           cv2.imshow(subjects[1], cv2.resize(predicted_img, (640, 480)))
#           cv2.waitKey(0)
#           cv2.destroyAllWindows()
#           cv2.waitKey(1)
#           cv2.destroyAllWindows()
#       elif keypressed == 'c':
#           cv2.destroyAllWindows()
print("Prediction complete")






