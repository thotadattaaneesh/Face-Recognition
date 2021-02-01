import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

info = np.load("Alr.npy")

print(info.shape, info.dtype)

X = info[:, 1:].astype(int)
y = info[:, 0]

MLmodel = KNeighborsClassifier()
MLmodel.fit(X, y)

cam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while(True):

    ret, Image = cam.read(0)

    if(ret):

        faces = classifier.detectMultiScale(Image)

        for i in faces:

            x, y, w, h = i

            rec = cv2.rectangle(Image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rec_win = Image[y:y+h, x:x+w]

            fix = cv2.resize(rec_win, (100, 100))
            opt = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)


            out = MLmodel.predict([opt.flatten()])

            print(out)

            cv2.imshow("Frame", opt)

        cv2.imshow("My Camera", Image)


    key = cv2.waitKey(1)

    if(key == ord("q") or key == ord("Q") or key == ord("E")):
        break

cam.release()
cv2.destroyAllWindows()