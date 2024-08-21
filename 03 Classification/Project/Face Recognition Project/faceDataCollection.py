import cv2
import numpy as np

# Creating the camera object (default laptop webcam)
cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier('/home/deadpool/Desktop/Machine-Learning-Projects/03 Classification/Project/Face Recognition Project/haarcascade_frontalface_alt.xml')

fileName = input('Enter Name : ')
dataset = './data/'
offset = 20
faceData = []
skip = 10

# Reading image from the camera object
while True:
    success, img = cam.read()
    if not success:
        print('No image found')
        break

    # Store the gray images
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face = model.detectMultiScale(img, scaleFactor=1.3, minNeighbors= 5)
    # pick the face with largest bounding box
    face = sorted(face, key=lambda f:f[2]*f[3])
    
    if len(face)>0:
        # pick the largest face
        face = face[-1]
        x,y,w,h = face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # crop and save the largest face
        crop_face = img[y-offset : y+h-offset, x-offset : x+w-offset]
        crop_face = cv2.resize(crop_face, (100,100))
        
        skip+=1
        if skip%10 == 0:
            faceData.append(crop_face)
            print(str(len(faceData))+' images of '+str(fileName)+' saved')

    cv2.imshow('image', img)
    # cv2.imshow('cropped face', crop_face)

    # Wait for 5000 ms (5 seconds) or until a key is pressed
    key = cv2.waitKey(1)

    # If the key 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# write the faceData on the disk
faceData = np.asarray(faceData)
print(faceData.shape)
n = faceData.shape[0]
faceData = faceData.reshape((n,-1))
print(faceData.shape)

# save the disk as np array
file_path = dataset + fileName + '.npy'
np.save(file_path,faceData)
print('Data of '+str(fileName)+' saved successfully : '+str(file_path))

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()