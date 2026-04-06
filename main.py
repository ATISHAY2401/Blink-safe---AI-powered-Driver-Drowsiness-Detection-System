#Drowsiness Detection System with Emergency Alert and Voice Recognition
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils
import pygame
import threading
import time
import speech_recognition as sr
import pywhatkit as kit
import geocoder
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pygame for sound
pygame.mixer.init()

# Load alarm sound
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except:
    logger.warning("Alarm sound file not found. Using default beep.")
    # Create a simple beep sound
    sample_rate = 44100
    duration = 1000  # milliseconds
    frequency = 440  # Hz
    n_samples = int(round(duration * 0.001 * sample_rate))
    buf = np.zeros((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        t = float(i) / sample_rate
        buf[i][0] = buf[i][1] = 0.5 * np.sin(2 * np.pi * frequency * t)
    alarm_sound = pygame.mixer.Sound(buffer=buf)

# Emergency contact number (with country code)
EMERGENCY_CONTACT = "+918303557744"

# Drowsiness detection parameters
EYE_AR_THRESH = 0.25
DROWSINESS_DETECTION_TIME = 3  # seconds before triggering alarm
EMERGENCY_ALERT_TIME = 7       # seconds before sending emergency message
COUNTER = 0
ALARM_ON = False
drowsiness_start_time = 0
alert_sent = False

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    logger.error("Shape predictor file not found. Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# Get the indices of facial landmarks for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def sound_alarm():
    global ALARM_ON
    # Play an alarm sound continuously until user wakes up
    while ALARM_ON:
        alarm_sound.play()
        time.sleep(3)  # Play for 3 seconds
        if not ALARM_ON:
            break
    alarm_sound.stop()

def get_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return f"Latitude: {g.lat}, Longitude: {g.lng}"
        else:
            return "Location not available"
    except Exception as e:
        logger.error(f"Error getting location: {e}")
        return "Location not available"

def send_whatsapp_alert(trigger_reason="Drowsiness detected"):
    global alert_sent
    try:
        location = get_location()
        current_time = datetime.now()
        message = f"EMERGENCY DETECTED!! {trigger_reason} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}. Location: {location}"
        
        # Get current time for scheduling the message
        now = datetime.now()
        hour = now.hour
        minute = now.minute + 1  # Send after 1 minute
        
        # Send WhatsApp message using pywhatkit
        kit.sendwhatmsg(EMERGENCY_CONTACT, message, hour, minute)
        
        alert_sent = True
        logger.info(f"WhatsApp alert sent to {EMERGENCY_CONTACT}")
        return True
    except Exception as e:
        logger.error(f"Error sending WhatsApp alert: {e}")
        return False

def voice_recognition():
    global ALARM_ON, alert_sent
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        logger.info("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Voice recognition active. Say 'emergency' for emergency alert.")
    
    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
            
            command = recognizer.recognize_google(audio).lower()
            logger.info(f"Recognized: {command}")
            
            if "emergency" in command:
                logger.info("Emergency command detected!")
                if not alert_sent:  # Prevent sending multiple alerts
                    if send_whatsapp_alert("User requested emergency via voice command"):
                        # Play confirmation sound
                        pygame.mixer.Sound(buffer=alarm_sound).play()
                        time.sleep(2)
        
        except sr.WaitTimeoutError:
            # No speech detected, continue listening
            pass
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except Exception as e:
            logger.error(f"Error in voice recognition: {e}")

def main():
    global COUNTER, ALARM_ON, drowsiness_start_time, alert_sent
    
    # Start voice recognition in a separate thread
    voice_thread = threading.Thread(target=voice_recognition, daemon=True)
    voice_thread.start()
    
    # Start video stream
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)  # Allow camera to warm up
    
    while True:
        ret, frame = vs.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
        
        # Resize frame and convert to grayscale
        frame = cv2.resize(frame, (450, 450))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        rects = detector(gray, 0)
        
        face_detected = False
        current_ear = 0
        
        for rect in rects:
            face_detected = True
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Calculate eye aspect ratio for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            # Average the eye aspect ratio
            ear = (leftEAR + rightEAR) / 2.0
            current_ear = ear
            
            # Compute the convex hull for each eye and draw it
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Check if eye aspect ratio is below threshold
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                
                # Start timer if this is the first frame of drowsiness
                if COUNTER == 1:
                    drowsiness_start_time = time.time()
                    alert_sent = False
                
                # Calculate how long the user has been drowsy
                drowsiness_duration = time.time() - drowsiness_start_time
                
                # If eyes are closed for 3 seconds, trigger alarm
                if drowsiness_duration >= DROWSINESS_DETECTION_TIME and not ALARM_ON:
                    ALARM_ON = True
                    # Sound alarm in a separate thread
                    threading.Thread(target=sound_alarm, daemon=True).start()
                    logger.info("Alarm triggered - user drowsy for 3 seconds")
                
                # If eyes are closed for 7 seconds, send emergency alert
                if drowsiness_duration >= EMERGENCY_ALERT_TIME and not alert_sent:
                    logger.info("Sending emergency alert - user drowsy for 7 seconds")
                    threading.Thread(target=send_whatsapp_alert, args=("Drowsiness detected for more than 7 seconds",), daemon=True).start()
                
                # Display alert on frame with timer
                cv2.putText(frame, f"DROWSY: {drowsiness_duration:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Reset if eyes are open
                if COUNTER > 0:
                    logger.info(f"User woke up after {time.time() - drowsiness_start_time:.1f} seconds")
                COUNTER = 0
                ALARM_ON = False
                drowsiness_start_time = 0
            
            # Display eye aspect ratio on frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # If no face detected, reset counters
        if not face_detected:
            COUNTER = 0
            ALARM_ON = False
            drowsiness_start_time = 0
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' key press
        if key == ord("q"):
            break
    
    # Cleanup
    ALARM_ON = False  # Stop alarm thread
    time.sleep(0.5)   # Give time for thread to stop
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()