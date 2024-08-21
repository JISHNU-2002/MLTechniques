import cv2
import os
import numpy as np

dataset = './data/'
faceData = []
labels = []
classId = 0
name2label = {}

for f in os.listdir(dataset):
    if f.endswith('.npy'):
        # X
        name2label[classId] = f[:-4]
        dataItem = np.load(dataset + f)
        # print(dataItem.shape)
        faceData.append(dataItem)
        n = dataItem.shape[0]
        
        # y
        target = classId * np.ones((n, ))
        classId+=1
        labels.append(target)
        
# print(faceData)
# print(labels)

XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels, axis=0).reshape(-1,1)
# print(X.shape, y.shape, name2label)



# Algrithm
def distance(p,q):
    return np.sqrt(np.sum(np.square(p-q)))

def knnClassifier(X,y,Xt,k=5):
  n = X.shape[0]
  dist = []

  for i in range(n):
    d = distance(X[i],Xt)
    dist.append((d,y[i]))

  dist  = sorted(dist, key=lambda x: x[0])
  # dist = np.array(dist[:k])
  # nearest = dist[:,1].astype(float)
  # Extract the labels of the k nearest neighbors
  nearest = np.array([label for _, label in dist[:k]])
  prediction = np.mean(nearest)

  return int(prediction)



# Prediction

# Creating the camera object (default laptop webcam)
cam = cv2.VideoCapture(0)
offset = 20
model = cv2.CascadeClassifier('/home/deadpool/Desktop/Machine-Learning-Projects/03 Classification/Project/Face Recognition Project/haarcascade_frontalface_alt.xml')

# Reading image from the camera object
while True:
    success, img = cam.read()
    if not success:
        print('No image found')
        break
    
    face = model.detectMultiScale(img, scaleFactor=1.3, minNeighbors= 5)
    
    # render a box around each image and prdict its name
    for f in face:
        x,y,w,h = f
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # crop and save the largest face
        crop_face = img[y-offset : y+h+offset, x-offset : x+w+offset]
        crop_face = cv2.resize(crop_face, (100,100))
        
        # prediction using knn
        Xt = crop_face.flatten()
        class_prediction = knnClassifier(XT, yT, Xt)
        name_prediction = name2label[class_prediction]
        
        # display the name and the box
        cv2.putText(img, name_prediction, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('predcition',img)
    # Wait for 5000 ms (5 seconds) or until a key is pressed
    key = cv2.waitKey(1)

    # If the key 'q' is pressed, exit the loop
    if key == ord('q'):
        break
