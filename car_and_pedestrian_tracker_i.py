# Made by KodeHak
import cv2

# Choose file for both image and classifier
img_file = 'Highway-Traffic.jpg'
classifier_file = 'cars.xml'

# Create opencv image
img = cv2.imread(img_file)

# Create pre-trained car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Must convert to grayscale(needed for haar cascade)
grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Detect cars
cars = car_tracker.detectMultiScale(grayscale_img)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the img with cars
cv2.imshow('KodeHak Car And Pedestrian Tracker', img)
cv2.waitKey()

print("Code Completed")