# Import kivy dependencies first
# Base app class
from kivy.app import App
# Box layer (layout that kivy app will take)
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
# Real-time webcam feed
from kivy.uix.image import Image
# Button
from kivy.uix.button import Button
# Text/label
from kivy.uix.label import Label

# Import other kivy stuff
# Allow us to make continuous updates (real-time feed)
from kivy.clock import Clock
# Convert image from OpenCV webcam to text jar and then set image equal to that
from kivy.graphics.texture import Texture
# See how app is performing without displaying such info to users
from kivy.logger import Logger

# Import other dependencies
# Import OpenCV for accessing webcam
import cv2
# Import TensorFlow
import tensorflow as tf
# Import custom cistance layer
from layers import L1Dist
# Easier to work with file path
import os 
# Import numpy
import numpy as np

# Build app and layout
# CamApp inherits behavior of App
class CamApp(App):

    def build(self):
        # Main layout components
        # Main image (image will take 80 percent of vertical height)
        self.web_cam = Image(size_hint=(1,.8))
        # Button says verify (full width but only 10 percent of height)
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        # Text that tells us whether verified or not (10 percent of height)
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        # Objects will appear in sequential order from top-down
        layout = BoxLayout(orientation='vertical')
        # Add image to layout
        layout.add_widget(self.web_cam)
        # Add button to layout
        layout.add_widget(self.button)
        # Add verification text to layout
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

        # Setup video capture device to connect to webcam
        self.capture = cv2.VideoCapture(0)
        # Run update function on this interval
        # Trigger update function in real-time
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        # Return value and frame as numpy array
        ret, frame = self.capture.read()
        # 950x950 frame
        frame = frame[150:1100, 550:1500, :]

        # Flip horizontal and convert image to texture so we can render image
        # in real-time

        # Converting OpenCV array which is an image into buffer and converting 
        # it to texture then rendering to webcam object

        # Flip image vertically and convert to byte buffer
        buf = cv2.flip(frame, 0).tobytes()
        # Create Kivy texture object iwth same dimensions as frame
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # Copies raw pixel bytes from OpenCV frame buf into Kivy's texture memory
        # Texture is what GPU uses to draw images on screen
        # This line is loading image data into texture so it can be displayed
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # Assign texture to Image widget so it displays on screen
        self.web_cam.texture = img_texture

    # Load image from file and convert to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load the image
        img = tf.io.decode_jpeg(byte_img)
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        # Scale image to be between 0 and 1
        img = img / 255.0
        # Return image
        return img
    
    # Verification function to verify person
    # frame - input image
    # model - trained siamese neural network
    # detection_threshold - metric above which prediction is considered positive
    # verification threshold - proportion of positive predictions / total positive samples
    def verify(self, *args):

        # Specify thresholds
        # Limit before prediction is considered positive
        detection_threshold = 0.5
        # What proportion of predictions need to be positive for a match
        verification_threshold = 0.5

        # Builds a file path to save our image from webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        # Captures image from webcam - true if camera succeeded + frame/image (NumPy array)
        ret, frame = self.capture.read()
        # 950x950 crop
        frame = frame[150:1100, 550:1500, :]
        # Save cropped image to save path
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = [] 
        # Loop through every image in verification_images folder
        for image in os.listdir(os.path.join('application_data', 'verification_images')):

            if not image.lower().endswith('.jpg'):
                continue
            
            # Grab input image from webcam and store it in input_image folder as input_image.jpg
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            # Grab verification (positive sample) image
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image)) 

            # Stack the two images
            # Insert a dimension so each image becomes a batch of size 1
            # Convert that into list
            # Feed those batched images into Siamese model
            # Have model compare embeddings and return similarity score
            result = self.model.predict([
                np.expand_dims(input_img, axis=0),
                np.expand_dims(validation_img, axis=0)])
            # Add to results array
            results.append(result)

        # Grabbing results and wrapping it in NumPy array
        # Summing up all examples that surpass detection_threshold
        # Detection threshold - determines how many positive predictions are passing detection_threshold
        detection = np.sum(np.array(results) > detection_threshold)
        # Verification threshold - proportion of positive predictions / total positive samples
        # What proportion of verification images matched input face?
        verification = detection / len(results)
        # If proportion is greater than verification threshold, person in webcam right now is verified
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details
        Logger.info(results)
        # Number of results that surpass detection threshold of 0.5
        Logger.info(detection)
        # What proportion of verification images matched input face?
        Logger.info(verification)
        # If proportion is greater than verification threshold, person in webcam right now is verified
        Logger.info(verified)

        # Number of results that surpass detection threshold
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))

        results_array = np.array(results).flatten()
        print("MIN:", np.min(results_array))
        print("MAX:", np.max(results_array))
        print("MEAN:", np.mean(results_array))

        return results, verified
    
if __name__ == '__main__':
    CamApp().run()