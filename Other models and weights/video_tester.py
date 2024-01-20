import cv2
import numpy as np
from keras.models import model_from_json
#load model
model = model_from_json(open("model.json","r").read())
#load weights
model.load_weights('model_weights.h5')
#loading the cascade classifier
detector = cv2.CascadeClassifier('C:\\Users\\arvin\\OneDrive\\Desktop\\Mini-Project\\haarcascade_frontalface_default.xml')
# Create a VideoCapture object
cap = cv2.VideoCapture(0)
while True:
    ret,test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray_img, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0, 255, 0),2)
        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        pred = model.predict(roi_gray[np.newaxis, :, :, np.newaxis])

        max_index = np.argmax(pred)
        emotions = ("Angry", "Disgust","Fear","Happy","Neutral","Sad","Surprise")
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion,(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000,700))
    cv2.imshow('Facial Emotion Detection', resized_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
