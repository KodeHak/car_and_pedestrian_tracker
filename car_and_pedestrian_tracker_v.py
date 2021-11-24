# Made by KodeHak
import cv2

# Choose file for video
video = cv2.VideoCapture('cp2.mp4')

# Choose file for both car and pedestrian classifier
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# Create pre-trained car and pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run forever until car stops or something
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale(needed for haar cascade)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        break

    # Detect cars AND pedestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    print(cars, "This is a car")
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)
    #print(pedestrians)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the video with cars and pedestrians
    cv2.imshow('KodeHak Car And Pedestrian Tracker', frame)
    key = cv2.waitKey(1)

    # top if the Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()

print("Code Completed")