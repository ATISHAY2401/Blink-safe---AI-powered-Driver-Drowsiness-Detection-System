🔹 1. Project Title

BlinkSafe – Driver Drowsiness Detection System

🔹 2. Project Description

BlinkSafe is an AI-based driver drowsiness detection system that monitors eye-blink patterns using computer vision. 
The system uses real-time face detection and Eye Aspect Ratio (EAR) calculation to identify signs of fatigue. 
If drowsiness is detected, an alert is triggered to prevent accidents. This project aims to enhance road safety using intelligent automation.

🔹 3. Features
1.Real-time face detection
2.Eye tracking using landmarks
3.EAR (Eye Aspect Ratio) calculation
4.Drowsiness detection logic
5.Alarm/alert system
6.Works with webcam/live video

🔹 4. Technologies Used
Python
OpenCV
Dlib / Mediapipe
NumPy
imutils

🔹 5. How It Works 

Start
  ↓
Webcam captures live video
  ↓
Face detection in each frame
  ↓
Eye landmarks extraction
  ↓
EAR (Eye Aspect Ratio) calculation
  ↓
Check: EAR < Threshold?
  ↓
Yes ──→ Check duration (continuous frames)
           ↓
        If yes → Drowsiness detected
           ↓
        Trigger alarm 🚨
  ↓
No ──→ Continue monitoring
  ↓
Repeat process


🔹 6. System Architecture / Modules

Module 1: Face Detection
Module 2: Eye Landmark Detection
Module 3: EAR Calculation
Module 4: Drowsiness Detection
Module 5: Alert System

🔹 7. Installation & Setup
git clone <your-repo-link>
cd blinksafe
pip install -r requirements.txt
python main.py

🔹 8. Requirements


1.dlib
2.numpy
3.imutils
4.scipy
5.Pywhatkit
6.Dlib
7.Matplotlib
8.Pandas
9.opencv-python
10.Face recognition 
11.voice recognition

🔹 9. Output / Results

The system successfully detects eye closure and triggers an alert when the driver appears drowsy.
It performs well under normal lighting conditions and can run in real-time.


🔹 10. Future Improvements
1.Mobile app integration
2.Night vision enhancement
3.Better accuracy using deep learning
4.Integration with car systems

🔹 11. Conclusion

BlinkSafe provides an effective and low-cost solution for detecting driver fatigue.
It demonstrates how AI and computer vision can be used to solve real-world safety problems.

🔹 12. Author

Atishay Bajpai
B.Tech Student – Axis Colleges
