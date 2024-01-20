import cv2
import numpy as np
import dlib
from keras.models import model_from_json
from statistics import mode
import os
import tkinter as tk
import pygame
from PIL import Image, ImageTk

model = model_from_json(open('model.json',"r").read())
model.load_weights('model_weights.h5')

detector = dlib.get_frontal_face_detector()
emotions = []
final_label = ""

def get_image():
    # Initialize video capture object
    cap = cv2.VideoCapture(0)
    # Loop to continuously capture frames
    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()
        # Display the frame
        cv2.imshow('frame', frame)
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        # If 'q' key is pressed, exit the loop
        if key == ord('q'):
            break
        # If 's' key is pressed, take a picture and process it
        elif key == ord('s'):
            # Convert image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image using dlib's detector
            faces = detector(gray)
            # Loop through each face in the image
            for face in faces:
                # Get the coordinates of the face region
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # Crop the face region from the image
                face_roi = gray[y:y+h, x:x+w]
                # Resize the face region to match the input size of the model
                face_roi = cv2.resize(face_roi, (48, 48))
                # Predict the emotion for the face using the pre-trained model
                emotion = model.predict(face_roi[np.newaxis, :, :, np.newaxis])
                # Get the index of the maximum value in the predicted emotion vector
                emotion_index = np.argmax(emotion)
                # Map the index to the corresponding emotion label
                if emotion_index == 0:
                    emotion_label = 'Angry'
                elif emotion_index == 1:
                    emotion_label = 'Disgust'
                elif emotion_index == 2:
                    emotion_label = 'Fear'
                elif emotion_index == 3:
                    emotion_label = 'Happy'
                elif emotion_index == 4:
                    emotion_label = 'Neutral'
                elif emotion_index == 5:
                    emotion_label = 'Sad'
                else:
                    emotion_label = 'Surprise'
                # Add the predicted emotion to the list
                emotions.append(emotion_label)
            # Display the captured image with bounding boxes around the detected faces
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Get the mode of the predicted emotions
            if emotions:
                mode_emotion = mode(emotions)
                # Draw the emotion label on the image
                cv2.putText(frame, mode_emotion, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Emotion Detection', frame)
            final_label = mode_emotion
            
            # Set up the Pygame mixer
            pygame.mixer.init()

            # Define the folder containing the songs
            songs_folder = "Songs/" + final_label

            # Create a list of all the songs in the folder
            song_files = os.listdir(songs_folder)

            # Initialize the current song index
            global current_song_index
            current_song_index=0
            # Function to play the current song
            def play_song():
                # Define the current song index
                current_song_index = 0
                # Load the current song file
                current_song_file = song_files[current_song_index]
                current_song_path = os.path.join(songs_folder, current_song_file)
                pygame.mixer.music.load(current_song_path)
                # Update the song name label
                song_name_label.config(text=current_song_file)
                # Play the song
                pygame.mixer.music.play()

            # Function to pause the current song
            def pause_song():
                pygame.mixer.music.pause()

            # Function to resume the current song
            def resume_song():
                pygame.mixer.music.unpause()

            # Function to play the next song
            def next_song():
                global current_song_index
                # Increment the current song index
                current_song_index += 1
                # Wrap around to the first song if necessary
                if current_song_index >= len(song_files):
                    current_song_index = 0
                # Stop the current song
                pygame.mixer.music.stop()
                # Load and play the next song
                current_song_file = song_files[current_song_index]
                current_song_path = os.path.join(songs_folder, current_song_file)
                pygame.mixer.music.load(current_song_path)
                # Update the song name label
                song_name_label.config(text=current_song_file)
                pygame.mixer.music.play()

            # Function to play the previous song
            def prev_song():
                global current_song_index
                # Decrement the current song index
                current_song_index -= 1
                # Wrap around to the last song if necessary
                if current_song_index < 0:
                    current_song_index = len(song_files) - 1
                # Stop the current song
                pygame.mixer.music.stop()
                # Load and play the previous song
                current_song_file = song_files[current_song_index]
                current_song_path = os.path.join(songs_folder, current_song_file)
                pygame.mixer.music.load(current_song_path)
                # Update the song name label
                song_name_label.config(text=current_song_file)
                pygame.mixer.music.play()

            # Create the main window
            window = tk.Tk()
            # window.geometry('400x350')
            window.title("Emotion based Music Player")

            # Create the song name label
            song_name_label = tk.Label(window, text="No song playing")
            song_name_label.pack()

            # create leftframe
            left_frame = tk.Frame(window)
            left_frame.pack(side="left", padx=10, pady=10)

            # Create the play button
            play_button = tk.Button(left_frame,text="Start", command=play_song)
            play_button.pack() 

            # Create the previous button
            file4 = Image.open("gui\prev.jpg").resize((30,30))
            buttonimg4 = ImageTk.PhotoImage(file4)
            prev_button = tk.Button(left_frame, image = buttonimg4, text="Previous", command=prev_song)
            prev_button.pack()

            # Create the pause button
            file2 = Image.open("gui\pause.jpg").resize((30,30))
            buttonimg2 = ImageTk.PhotoImage(file2)
            pause_button = tk.Button(left_frame, image = buttonimg2, text="Pause", command=pause_song)
            pause_button.pack()

            # Create the resume button
            file1 = Image.open("gui\play.jpg").resize((30,30))
            buttonimg1 = ImageTk.PhotoImage(file1)
            resume_button = tk.Button(left_frame, image = buttonimg1, text="Resume", command=resume_song)
            resume_button.pack()

            # Create the next button
            file3 = Image.open("gui\\next.jpg").resize((30,30))
            buttonimg3 = ImageTk.PhotoImage(file3)
            next_button = tk.Button(left_frame, image = buttonimg3, text="Next", command=next_song)
            next_button.pack()

            # Create a frame for the right side of the window
            right_frame = tk.Frame(window)
            right_frame.pack(side="right", padx=10, pady=10)

            # Create the listbox widget
            song_listbox = tk.Listbox(right_frame)
            song_listbox.pack()

            # Populate the listbox with the song names
            for song_file in song_files:
                song_listbox.insert(tk.END, song_file)

            # Highlight the currently playing song
            def highlight_current_song():
                current_song_index = pygame.mixer.music.get_busy()
                if current_song_index != 0:
                    song_listbox.selection_clear(0, tk.END)
                    song_listbox.selection_set(current_song_index - 1)

            # Add a callback function to update the song name label and highlight the currently playing song
            def update_ui():
                highlight_current_song()
                current_song_file = song_files[current_song_index]
                song_name_label.config(text=current_song_file)

            # Call the update_ui function periodically
            window.after(100, update_ui)


            # Start the main loop
            window.mainloop()
            emotions.clear()
    # Release the video capture object and close all windows
    cap.release()

get_image()
