import cv2

# Creating the camera object (default laptop webcam)
cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier('/home/deadpool/Desktop/Machine-Learning-Projects/03 Classification/Project/Face Recognition Project/haarcascade_frontalface_alt.xml')

# Reading image from the camera object
while True:
    success, img = cam.read()
    if not success:
        print('No image found')
        break

    face = model.detectMultiScale(img, scaleFactor=1.2, minNeighbors= 3)
    for f in face:
      x,y,w,h = f
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('image', img)

    # Wait for 5000 ms (5 seconds) or until a key is pressed
    key = cv2.waitKey(1)

    # If the key 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()