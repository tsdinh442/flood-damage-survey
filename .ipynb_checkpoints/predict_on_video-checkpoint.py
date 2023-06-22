import cv2
import numpy as np
import joblib
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


SIZE = 256  # Resize images
RED = (0, 0, 255, 100)
GREEN = (0, 255, 0, 100)


# Define the text and its properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 255, 255)  # Blue color in BGR format
thickness = 2


# Load the saved model from the file
with open('./models/2/XGBoost.pkl', 'rb') as file:
    model = pickle.load(file)
with open('./models/2/vgg.pkl', 'rb') as file:
    VGG = pickle.load(file)
Seq = tf.keras.models.load_model('models/2/SEQ.h5')

def predict(model, data):

    data = data / 255
    input_img = np.expand_dims(data, axis=0)
    feature_extractor = VGG.predict(input_img)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    pred = model.predict(features)[0]

    return pred

SIZE = 224




# Read the video
video_path = 'BusyParkingLotUAVVideo.mp4'
cap = cv2.VideoCapture(video_path)

# Read the mask file (already black and white)
mask = cv2.imread('frame_2s.png', 0)

# Find contours in the mask image
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ret = True
#step = 30
frame_count = 0
while ret:

    ret, frame = cap.read()
    if frame_count > 47:
        #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        #img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 25)

        #_mask = np.zeros_like(frame, dtype=np.uint8)
        overlay = frame.copy()
        if frame_count % 12 == 0:
            EMPTY = []
            OCCUPIED = []
            stalls = []
            # Iterate over each contour
            for i, contour in enumerate(contours):
                # Create an empty mask for the contour
                contour_mask = np.zeros_like(mask)

                # Draw the current contour on the mask
                cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

                # Bitwise AND operation to crop the image using the contour mask
                cropped = cv2.bitwise_and(frame, cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR))
                # Find the bounding rectangle for the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the region from the original image using the bounding rectangle
                cropped_mask = cropped[y:y + h, x:x + w]
                cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_RGB2BGR)
                resized = cv2.resize(cropped_mask, (SIZE, SIZE))
                #resized = resized / 255
                input_data = np.expand_dims(resized, axis=0)
                predictions = Seq.predict(preprocess_input(input_data))
                #print(predictions)
                predicted_labels = np.where(predictions > 0.5, 1, 0)
                print(predicted_labels)

                if predicted_labels[0][0] == 0:
                    EMPTY.append(contour)
                else:
                    OCCUPIED.append(contour)
            # color the stall with according colors


        colored_image = cv2.drawContours(frame, EMPTY, -1, GREEN, thickness=cv2.FILLED)
        colored_image = cv2.drawContours(colored_image, OCCUPIED, -1, RED, thickness=cv2.FILLED)

        alpha = 0.6
        frame_new = cv2.addWeighted(overlay, alpha, colored_image, 1 - alpha, 0)

        # Put the text on the frame
        cv2.rectangle(frame_new, (50, 100), (50 + 150, 100 - 20), (255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(frame_new, 'Available: ' + str(len(EMPTY)), (50 + 10, 100 - 7), font, font_scale, (0, 0, 0), thickness)
        cv2.rectangle(frame_new, (50, 130), (50 + 150, 130 - 20), (255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(frame_new, 'Occupied: ' + str(len(OCCUPIED)), (50 + 10, 130 - 7), font, font_scale, (0, 0, 0),
                    thickness)

        cv2.imwrite('./frames/6/{}.jpg'.format(frame_count), frame_new)
    frame_count += 1

    # Display or save the cropped image
    #cv2.imshow('Cropped Image', resized)
    #cv2.waitKey(0)

    # Save the cropped image
    #video_writer.write(frame)
    if frame_count == 500:
        break

cap.release()
cv2.destroyAllWindows()



