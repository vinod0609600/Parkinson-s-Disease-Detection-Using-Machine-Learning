import sys
import os
import pyaudio
import wave
import numpy as np
import librosa
import joblib
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel,
                             QComboBox, QFormLayout, QGroupBox, QScrollArea, QFrame, QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont, QColor, QPalette

# Load model and scaler
svm_model = joblib.load('svm_parkinsons_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# Simulated predefined patient data
predefined_patients = {
    "Patient 1": [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
        0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
        0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
    ]
}

def predict_from_features(features_df):
    try:
        scaled = scaler.transform(features_df)
        prediction = svm_model.predict(scaled)
        return prediction[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def extract_features(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
        if len(y) < 2048: return None
        y = librosa.to_mono(y)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        pitch, mag = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitch[pitch > 0])
        jitter = np.std(pitch[pitch > 0])
        shimmer = np.std(librosa.feature.rms(y=y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)[:3]
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)[:2]
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features = np.hstack([mfcc, [pitch_mean, jitter, shimmer], chroma, spectral_contrast, [zcr]])
        return pd.DataFrame([features], columns=feature_names)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

class ParkinsonsDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hospital Parkinson's Dashboard")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("font-family: Arial;")

        # Set hospital-themed background color
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#e6f2ff"))  # Light hospital blue
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.layout = QVBoxLayout(self)

        self.create_header()
        self.create_tabs()
        self.create_footer()

    def create_header(self):
        header = QLabel("Parkinson's Disease Predictor")
        header.setFont(QFont("Arial", 22, QFont.Bold))
        header.setStyleSheet("color: #003366; padding: 15px;")
        header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header)

    def create_tabs(self):
        tabs = QTabWidget()

        # **Tabs Section**
        self.main_tab = QWidget()
        self.help_tab = QWidget()
        self.about_tab = QWidget()

        tabs.addTab(self.main_tab, "Main Dashboard")
        tabs.addTab(self.help_tab, "Help Info")
        tabs.addTab(self.about_tab, "About Disease")

        # Main Dashboard Tab
        self.main_layout = QVBoxLayout()
        self.create_main_area()
        self.main_tab.setLayout(self.main_layout)

        # Help Info Tab
        self.help_layout = QVBoxLayout()
        help_label = QLabel("For assistance, contact us at: hospital@example.com")
        help_label.setFont(QFont("Arial", 14))
        help_label.setStyleSheet("color: #003366; padding: 15px;")
        self.help_layout.addWidget(help_label)
        self.help_tab.setLayout(self.help_layout)

        # About Disease Tab
        self.about_layout = QVBoxLayout()
        about_label = QLabel("Parkinson's disease is a progressive nervous system disorder that affects movement...")
        about_label.setFont(QFont("Arial", 14))
        about_label.setStyleSheet("color: #003366; padding: 15px;")
        self.about_layout.addWidget(about_label)
        self.about_tab.setLayout(self.about_layout)

        self.layout.addWidget(tabs)

    def create_main_area(self):
        container = QHBoxLayout()

        # Left Section (Inputs)
        self.input_section = QVBoxLayout()

        self.record_btn = QPushButton("🎙️ Start Recording")
        self.record_btn.clicked.connect(self.handle_recording)
        self.record_btn.setStyleSheet("background-color: #66ccff; font-weight: bold; padding: 10px;")
        self.input_section.addWidget(self.record_btn)

        self.manual_btn = QPushButton("📝 Enter Details Manually")
        self.manual_btn.clicked.connect(self.show_manual_inputs)
        self.manual_btn.setStyleSheet("background-color: #99e6e6; font-weight: bold; padding: 10px;")
        self.input_section.addWidget(self.manual_btn)

        self.input_group = QGroupBox("Manual Feature Input")
        self.input_form = QFormLayout()
        self.entries = []

        for feature in feature_names:
            line = QLineEdit()
            line.setPlaceholderText(feature)
            self.entries.append(line)
            self.input_form.addRow(feature, line)

        self.input_group.setLayout(self.input_form)
        self.input_section.addWidget(self.input_group)

        self.dropdown = QComboBox()
        self.dropdown.addItem("Select Patient")
        for patient in predefined_patients:
            self.dropdown.addItem(patient)
        self.dropdown.currentIndexChanged.connect(self.autofill_predefined)
        self.input_section.addWidget(self.dropdown)

        self.predict_btn = QPushButton("🧠 Predict")
        self.predict_btn.clicked.connect(self.predict_manual_input)
        self.predict_btn.setStyleSheet("background-color: #4dc3ff; font-weight: bold;")
        self.input_section.addWidget(self.predict_btn)

        self.clear_btn = QPushButton("❌ Clear All")
        self.clear_btn.clicked.connect(self.clear_inputs)
        self.clear_btn.setStyleSheet("background-color: #ff9999; font-weight: bold;")
        self.input_section.addWidget(self.clear_btn)

        # Add a hide features button
        self.hide_features_btn = QPushButton("❌ Hide Features")
        self.hide_features_btn.clicked.connect(self.hide_manual_inputs)
        self.hide_features_btn.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        self.input_section.addWidget(self.hide_features_btn)

        container.addLayout(self.input_section, 2)

        # Right Section (Output)
        self.result_section = QVBoxLayout()
        self.result_label = QLabel("Prediction result will appear here")
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #003366; border: 1px solid #66ccff; padding: 20px; background-color: white;")
        self.result_section.addWidget(self.result_label)
        container.addLayout(self.result_section, 3)

        self.main_layout.addLayout(container)

    def create_footer(self):
        footer = QLabel("Contact Us: hospital@example.com | +91-99999-99999")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; padding: 10px; font-size: 12px; border-top: 1px solid #ccc;")
        self.layout.addWidget(footer)

    def handle_recording(self):
        self.record_btn.setEnabled(False)
        self.manual_btn.setEnabled(False)
        self.result_label.setText("Recording... Please wait")
        QApplication.processEvents()

        filename = self.record_audio(5)
        if not filename:
            self.result_label.setText("Recording Failed")
            self.record_btn.setEnabled(True)
            self.manual_btn.setEnabled(True)
            return

        features = extract_features(filename)
        if features is not None:
            prediction = predict_from_features(features)
            self.display_prediction(prediction)
        else:
            self.result_label.setText("Error in feature extraction")
        self.record_btn.setEnabled(True)
        self.manual_btn.setEnabled(True)

    def record_audio(self, duration=5, filename="audio_sample.wav"):
        import time
        p = pyaudio.PyAudio()
        chunk, format, channels, rate = 1024, pyaudio.paInt16, 1, 44100
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        frames = [stream.read(chunk) for _ in range(int(rate / chunk * duration))]
        stream.stop_stream()
        stream.close()
        p.terminate()
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        return filename if os.path.exists(filename) else None

    def show_manual_inputs(self):
        self.input_group.show()
        self.dropdown.show()
        self.predict_btn.show()
        self.clear_btn.show()
        self.hide_features_btn.show()

    def hide_manual_inputs(self):
        self.input_group.hide()
        self.dropdown.hide()
        self.predict_btn.hide()
        self.clear_btn.hide()
        self.hide_features_btn.hide()

    def autofill_predefined(self):
        patient = self.dropdown.currentText()
        if patient in predefined_patients:
            values = predefined_patients[patient]
            for i in range(len(self.entries)):
                self.entries[i].setText(str(values[i]))

    def predict_manual_input(self):
        try:
            values = [float(entry.text()) for entry in self.entries]
            df = pd.DataFrame([values], columns=feature_names)
            prediction = predict_from_features(df)
            self.display_prediction(prediction)
        except ValueError:
            self.result_label.setText("Please enter valid numeric values.")

    def clear_inputs(self):
        for entry in self.entries:
            entry.clear()
        self.result_label.setText("Cleared. Ready for input.")

    def display_prediction(self, prediction):
        if prediction == 1:
            self.result_label.setText("🚨 Parkinson's Disease Detected")
            self.result_label.setStyleSheet("color: red; font-weight: bold; background-color: white; padding: 20px;")
        else:
            self.result_label.setText("✅ No Parkinson's Disease Detected")
            self.result_label.setStyleSheet("color: green; font-weight: bold; background-color: white; padding: 20px;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ParkinsonsDashboard()
    window.show()
    sys.exit(app.exec_())



# import sys
# import os
# import pyaudio
# import wave
# import numpy as np
# import librosa
# import joblib
# import pandas as pd
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QFont
# from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFormLayout, QLineEdit, QGroupBox, QComboBox, QFrame, QSpacerItem, QSizePolicy


# # Dummy feature names and predefined patients for demonstration
# feature_names = [
#     "MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "NHR", "RPDE",
#     "MDVP:Shimmer(dB)", "MDVP:Jitter(dB)", "Shimmer(dB)", "NHR(dB)", "MDVP:Fo", "MDVP:Jitter", "Shimmer",
#     "HNR", "RPDE:Fo", "RPDE:Jitter", "RPDE:Shimmer", "RPDE:NHR", "Fo", "Jitter", "Shimmer", "HNR", "RPDE"
# ]
# predefined_patients = ["Patient 1", "Patient 2", "Patient 3"]

# class ParkinsonsDashboard(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Parkinson's Disease Prediction")
#         self.setGeometry(100, 100, 900, 600)
#         self.main_layout = QVBoxLayout()

#         self.set_light_theme()
#         self.create_main_area()

#         container = QFrame()
#         container.setLayout(self.main_layout)
#         self.setCentralWidget(container)

#     def set_light_theme(self):
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #f0f0f0;
#             }
#             QPushButton {
#                 background-color: #66ccff;
#                 color: white;
#                 padding: 10px;
#                 border-radius: 5px;
#                 font-size: 16px;
#             }
#             QPushButton:hover {
#                 background-color: #3399cc;
#             }
#             QLabel {
#                 font-size: 18px;
#                 font-weight: bold;
#             }
#         """)

#     def create_main_area(self):
#         container = QHBoxLayout()

#         # Right Section (Output)
#         self.result_section = QVBoxLayout()

#         # Initialize the result label earlier
#         self.result_label = QLabel("Prediction result will appear here")
#         self.result_label.setFont(QFont("Arial", 16))
#         self.result_label.setAlignment(Qt.AlignCenter)
#         self.result_label.setStyleSheet("color: #003366; border: 1px solid #66ccff; padding: 20px; background-color: white;")
#         self.result_section.addWidget(self.result_label)

#         # Left Section (Inputs)
#         self.input_section = QVBoxLayout()

#         self.record_btn = QPushButton("🎙️ Start Recording")
#         self.record_btn.clicked.connect(self.handle_recording)
#         self.record_btn.setStyleSheet("background-color: #66ccff; font-weight: bold; padding: 10px;")
#         self.input_section.addWidget(self.record_btn)

#         self.manual_btn = QPushButton("📝 Enter Details Manually")
#         self.manual_btn.clicked.connect(self.hide_manual_inputs)  # Change this to hide_manual_inputs
#         self.manual_btn.setStyleSheet("background-color: #99e6e6; font-weight: bold; padding: 10px;")
#         self.input_section.addWidget(self.manual_btn)

#         self.input_group = QGroupBox("Manual Feature Input")
#         self.input_form = QFormLayout()
#         self.entries = []

#         for feature in feature_names:
#             line = QLineEdit()
#             line.setPlaceholderText(feature)
#             self.entries.append(line)
#             self.input_form.addRow(feature, line)

#         self.input_group.setLayout(self.input_form)
#         self.input_section.addWidget(self.input_group)

#         self.dropdown = QComboBox()
#         self.dropdown.addItem("Select Patient")
#         for patient in predefined_patients:
#             self.dropdown.addItem(patient)
#         self.dropdown.currentIndexChanged.connect(self.autofill_predefined)
#         self.input_section.addWidget(self.dropdown)

#         self.predict_btn = QPushButton("🧠 Predict")
#         self.predict_btn.clicked.connect(self.predict_manual_input)
#         self.predict_btn.setStyleSheet("background-color: #4dc3ff; font-weight: bold;")
#         self.input_section.addWidget(self.predict_btn)

#         self.clear_btn = QPushButton("❌ Clear All")
#         self.clear_btn.clicked.connect(self.clear_inputs)
#         self.clear_btn.setStyleSheet("background-color: #ff9999; font-weight: bold;")
#         self.input_section.addWidget(self.clear_btn)

#         # Add a hide features button
#         self.hide_features_btn = QPushButton("❌ Hide Features")
#         self.hide_features_btn.clicked.connect(self.hide_manual_inputs)
#         self.hide_features_btn.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
#         self.input_section.addWidget(self.hide_features_btn)

#         container.addLayout(self.input_section, 2)

#         self.main_layout.addLayout(container)

#     def hide_manual_inputs(self):
#         # Hide the manual inputs and change button text
#         for entry in self.entries:
#             entry.setVisible(not entry.isVisible())
#         if self.entries[0].isVisible():
#             self.hide_features_btn.setText("❌ Hide Features")
#         else:
#             self.hide_features_btn.setText("📝 Show Features")

#     def autofill_predefined(self):
#         # Dummy method to autofill based on selected patient
#         selected_patient = self.dropdown.currentText()
#         if selected_patient != "Select Patient":
#             for entry in self.entries:
#                 entry.setText(f"Value for {selected_patient}")
    
#     def handle_recording(self):
#         # Placeholder for handle recording
#         print("Recording started...")

#     def predict_manual_input(self):
#         # Placeholder for prediction logic
#         print("Prediction in progress...")
#         self.result_label.setText("Prediction Result: Positive")
    
#     def clear_inputs(self):
#         # Clears all the inputs and resets result label
#         for entry in self.entries:
#             entry.clear()
#         self.result_label.setText("Prediction result will appear here")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ParkinsonsDashboard()
#     window.show()
#     sys.exit(app.exec_())
