import cv2
import os
import face_recognition

import urllib.request #para abrir y leer URL
import numpy as np

# Codificar los rostros extraidos
#imageFacesPath = "C:/Users/Desktop/faceRecognition2/images/faces"
imageFacesPath = "C:/Users/gmbln/Desktop/ESP32-CAM-wireless-computer-vision-objects-detection-main/faces"
facesEncodings = []
facesNames = []
for file_name in os.listdir(imageFacesPath):
     image = cv2.imread(imageFacesPath + "/" + file_name)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
     facesEncodings.append(f_coding)
     facesNames.append(file_name.split(".")[0])
#print(facesEncodings)
#print(facesNames)
##############################################


# LEYENDO VIDEO
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

url = "http://192.168.0.140/cam-lo.jpg"
img_resp=urllib.request.urlopen(url)
imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
img=cv2.imdecode(imgnp,-1)
imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

cap = imgS

# Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
     ret, frame = cap.read()
     if ret == False:
          break
     frame = cv2.flip(frame, 1)
     orig = frame.copy()
     faces = faceClassif.detectMultiScale(frame, 1.1, 5)

     for (x, y, w, h) in faces:
          face = orig[y:y + h, x:x + w]
          face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
          actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
          result = face_recognition.compare_faces(facesEncodings, actual_face_encoding)
          #print(result)
          if True in result:
               index = result.index(True)
               name = facesNames[index]
               color = (125, 220, 0)
          else:
               name = "Desconocido"
               color = (50, 50, 255)
          cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
          cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
          cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
     cv2.imshow("Frame", frame)
     k = cv2.waitKey(1) & 0xFF
     if k == 27:
          break
cap.release()
cv2.destroyAllWindows()