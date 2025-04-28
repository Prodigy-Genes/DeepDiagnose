# DeepDiagnose 🩺

**Multi-Disease X-ray Classifier**  
A Streamlit-powered web app and training pipeline that automatically detects whether an uploaded X-ray is a chest or joint image, then runs the appropriate TensorFlow model to predict Pneumonia (97%+ accuracy) or Osteoarthritis (91%+ accuracy). Includes feedback capture for continual fine-tuning.

---

## 🚀 Features

- **Automatic X-ray type detection** (Chest vs Joint)  
- **Pneumonia classifier** (binary CNN, label smoothing + augmentation)  
- **Osteoarthritis classifier** (EfficientNetB0 transfer learning)  
- **Interactive Streamlit UI** with upload, prediction, confidence gauge  
- **Image validation filter** rejects non–X-ray inputs  
- **User feedback logging** for model retraining  
- **Extensible architecture** for future diseases

---

## 📁 Repository Structure

```text
DeepDiagnose/
├── backend/
│   ├── app.py                   # Streamlit front-end
|   ├── fine_tune.py             # For future fine-tuning of models using feedbacks
│   └── feedback/                # Captured user corrections
│       ├── pneu/
│       ├── osteo/
│       └── normal/
├── models/
│   ├── pneumonia_classifier.keras
│   └── osteo_efficientnetb0.keras
|   ├── anatomical_classifer.keras
├── test/
│   ├── pneumonia
│   └── Osteoarthritis
├── requirements.txt             # All Python dependencies
└── README.md


🛠️ Installation
Clone the repo
git clone https://github.com/your-username/DeepDiagnose.git
cd DeepDiagnose

Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
.\venv\Scripts\activate       # Windows

Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

(Optional) GPU Support
If you have a GPU, install tensorflow-gpu instead of tensorflow.



🖥️ Running the App
Ensure models are in place under models/ with the correct names.

Launch Streamlit:
streamlit run backend/app.py

Open the local URL printed in your console (e.g. http://localhost:8501).

Upload up to 5 X-ray images; the app will:

Auto-detect Chest vs Joint.

Run the appropriate model.

Display prediction, confidence, and allow corrections.

🔄 Feedback & Fine-Tuning
Any “No—Actual” corrections are saved under backend/feedback/{pneu,osteo,normal}.

A log feedback_log.csv keeps track of filename, predicted vs corrected labels.

Future training runs can include this feedback data to continually improve accuracy.



🤝 Contributing
Fork this repository.

Create a feature branch: git checkout -b feature/YourFeature.

Commit your changes: git commit -m "Add awesome feature".

Push to your fork: git push origin feature/YourFeature.

Open a pull request here.

Please make sure new code is well-documented and includes any necessary tests.

📝 License
This project is licensed under the MIT License. 

Built with ❤️ by prodigygenes — keep diagnosing!


