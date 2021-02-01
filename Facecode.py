import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

input = input("enter the name: ")

frames = []
outputs = []
while(True):

    ret, Image = cam.read(0)

    if (ret):
        faces = classifier.detectMultiScale(Image)

        for i in faces:

            x, y, w, h = i

            rec = cv2.rectangle(Image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rec_win = Image[y:y + h, x:x + w]
            fix = cv2.resize(rec_win, (100, 100))
            opt = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        cv2.imshow("My Camera", Image)
        cv2.imshow("Frame", opt)

    key = cv2.waitKey(1)

    if(key == ord("q") or key == ord("Q") or key == ord("E")):
        break

    if (key == ord("c") or key == ord("C")):
        #cv2.imwrite("cam.jpg", Image)
        frames.append(opt.flatten())
        outputs.append([input])


hly = np.array(frames)

vly = np.array(outputs)
print(type(vly))

save = np.hstack([vly, hly])
print(save.shape)
print(outputs)

exist = "Alr.npy"
if(os.path.exists(exist)):
    old = np.load(exist)
    save = np.vstack([old, save])

np.save(exist, save)

cam.release()
cv2.destroyAllWindows()