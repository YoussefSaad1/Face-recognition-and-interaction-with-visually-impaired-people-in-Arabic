import face_recognition
import cv2
import numpy as np
import pandas as pd
#google text to speech lirary
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import os 

##########################################################################
#--------------------------Database-------------------------------------
## This section represents the format which data should be saved  

# Load a sample picture and learn how to recognize it.
#image1 = face_recognition.load_image_file("joe.jpg")
image2 = face_recognition.load_image_file("image1.jpg")
image3 = face_recognition.load_image_file("image2.jpg")
images = [image2, image3]
names_data = ['person1','person2']
data = {'images':images,'names':names_data}

#whatever database or any data structure you use data should be returned as dataframe  
Data_base = pd.DataFrame(data)

# This function represents the speech to text part 
def listen_from_me():
    r = sr.Recognizer()     
    r.energy_threshold = 4000                                                                         
    with sr.Microphone() as source:                                                                       
        print("Speak:")                                                                                   
        audio = r.listen(source)
        r.pause_threshold = 3
    return audio 

# This function represents the text to speech part 
def say_to_me(text_speeched):
    language = 'ar'
    obj = gTTS(text=text_speeched, lang=language, slow=False)
    obj.save("audio.mp3")
    # Playing the converted file
    playsound('audio.mp3')
    os.remove("audio.mp3")
        
known_face_encodings = []
known_face_names = []
for img,names in zip(Data_base['images'], Data_base['names']):
    # Create arrays of known face encodings and their names
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    known_face_names.append(names)
    
    
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#########################################################################
#------------------------image to recognize-------------------------------

#1. Capture image from webcam

video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = video_capture.read()
# Close device
video_capture.release()

from matplotlib import pyplot as plt
frameRGB = frame[:,:,::-1] # BGR => RGB
plt.imshow(frameRGB)
###################################################################

######################################################################
#2. if you want to load you own image 
'''
path = ''
frame = cv2.imread(path)
'''
###########################################################################

# Resize frame of video to 1/4 size for faster face recognition processing
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_small_frame = small_frame[:, :, ::-1]
    
# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
name = "Unknown"

# If a match was found in known_face_encodings, just use the first one.
    
face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
best_match_index = np.argmin(face_distances)


if matches[best_match_index]:
    name = known_face_names[best_match_index]
else:
    # text to be converted to speech
    text_speeched = 'لم نستطع التعرف علي هذا الشخص هل عرفته نعم ام لا  '
    say_to_me(text_speeched) # say the text 
    
    audio = listen_from_me() # listen from user 
    r = sr.Recognizer()
    try:
        said = r.recognize_google(audio, language="ar")
        print('you said' + said)
        if (said == 'نعم'):
            text_speeched = 'من فضلك عرفنا بلاسم '
            say_to_me(text_speeched)
            audio2 = listen_from_me()
            name = r.recognize_google(audio2, language="ar")
            print('name you said'+name)
            images.append(frame)
            names_data.append(name)
            data = {'images':images,'names':names}
            Data_base = pd.DataFrame(data)
            text_speeched = 'تمت اضافته في قاعدة البيانات بنجاح '
        elif(said == 'لا'):
            name = "Unknown"
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    
face_names.append(name)
if (name == "Unknown" ):
    text_speeched = 'شخص غير معروف '
    say_to_me(text_speeched)
else:
    
    text_speeched =  "قل ل"+name + " اهلا" 
    say_to_me(text_speeched)
    




