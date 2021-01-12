from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

model = load_model("mask_detector/mask_detector.h5")
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y-10:y+h+10, x-10:x+w+10]
        roi_color = frame[y-10:y+h+10, x-10:x+w+10]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi_color, (224, 224))
        roi = preprocess_input(img_to_array(roi))
        roi = np.expand_dims(roi, 0)
        
        output = model.predict(roi)
        i = np.argmax(output)
        if i==0:
            confidence = output[0][i]*100
            output = 'Mask'
            s = output + ' ' + "{:.2f}".format(confidence) + '%'
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
            frame = cv2.putText(frame, s, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if i==1:
            confidence = output[0][i]*100
            output = 'No Mask'
            s = output + ' ' + "{:.2f}".format(confidence) + '%'
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), 2)
            frame = cv2.putText(frame, s, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # print(output, confidence)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
