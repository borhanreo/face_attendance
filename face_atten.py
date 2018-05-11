import argparse

import cv2
import os
import numpy as np
subjects = ["", "Borhan","Tawhid","Arif_iOS"]
file = open('train_data.txt', 'w')
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True,
	help="path to Caffe 'deploy' prototxt file")
args = vars(ap.parse_args())
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    index = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            #print (image_path)
            image = cv2.imread(image_path)
            print (image)
            file.write(image)
            print ("...processing....")

            #small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            #height, width, channels = small.shape
            #cv2.imshow("Training on image...", cv2.resize(small, (width, height), fx=0.5, fy=0.5))
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(1)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    file.close()
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

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img
print("Predicting images...")

# load test images
# test_img1 = cv2.imread("test-data/test1.jpg")
# test_img2 = cv2.imread("test-data/test2.jpg")
# test_img3 = cv2.imread("test-data/test3.jpg")
# test_img3 = cv2.imread("test-data/test3.jpg")
test_img4 = cv2.imread(args["image"])

# perform a prediction
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
print("Prediction complete")

# display both images
# cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
# cv2.imshow(subjects[3], c1v2.resize(predicted_img3, (400, 500)))

while True:
    print('Your score so far is {}.'.format(myScore))
    print("Would you like to roll or quit?")
    ans = input("Roll...")
    if ans.lower() == 'r':
        R = np.random.randint(1, 8)
        print("You rolled a {}.".format(R))
        myScore = R + myScore
    else:
        print("Now I'll see if I can break your score...")
        break

cv2.imshow(subjects[3], cv2.resize(predicted_img4, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()





