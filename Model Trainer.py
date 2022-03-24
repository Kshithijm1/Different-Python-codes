import cv2
import numpy as np
from PIL import Image
import os

path = "C:/Users/kshit/.conda/samples"

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
FaceCascade = cv2.CascadeClassifier(cascPath)

def Images_And_Labels(path):
    ImagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    for ImagePath in ImagePaths:
    
        gray_img = Image.open(ImagePath).convert('L')
        img_arr = np.array(gray_img, 'uint8')
    
        id = int(os.path.split(ImagePath)[-1].split(".")[1])
        faces = faces = FaceCascade.detectMultiScale(img_arr)
    
        for (x,y,w,h) in faces:
            faceSamples.append(img_arr[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print("Training faces, It will take a few seconds. Wait...")
faces, ids = Images_And_Labels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('C:/Users/kshit/.conda/trainer/trainer.yml')

print("Model trained, Now we can recognize your face")
