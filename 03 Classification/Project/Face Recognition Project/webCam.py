import cv2

# Creating the camera object (default laptop webcam)
cam = cv2.VideoCapture(0)

# Reading image from the camera object
while True:
    success, img = cam.read()
    if not success:
        print('No image found')
        break

    cv2.imshow('image', img)

    # Wait for 1 ms or until a key is pressed
    key = cv2.waitKey(1)

    # If the key 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
