🥔 Potato Disease Classification System
🌿 Overview

This project is a Convolutional Neural Network (CNN) based system that classifies potato leaf diseases into three categories — Early Blight, Late Blight, and Healthy.
The model is trained using TensorFlow/Keras, achieving 96.8% validation accuracy, and deployed through an interactive Streamlit web application for real-time inference.

🧠 Features

✅ CNN-based deep learning model trained on potato leaf images

🌱 Classifies: Healthy, Early Blight, Late Blight

🔁 Image augmentation for robust training

⚡ Streamlit web interface for real-time image uploads and predictions

🧩 Fast inference — average latency ≈ 120 ms

💾 Includes trained model (potatoes.keras) and sample test images

🏗️ Project Structure
potato-disease-classification/
│
├── api/                         # Optional API files
├── saved_models/                # Saved model checkpoints
├── test_images_from_internet/   # Sample test images
├── training/                    # Training notebooks & scripts
├── app.py                       # Streamlit app for prediction
├── potatoes.keras                # Final trained model
├── requirements.txt             # Dependencies
├── .gitignore
└── README.md                    # Project documentation

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/ashokmishra1234/potato-disease-classification.git
cd potato-disease-classification

2. Create a virtual environment
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows

3. Install dependencies
pip install -r requirements.txt

🚀 Usage
Run the Streamlit Web App
streamlit run app.py

Upload a Leaf Image

Click on the “Choose a leaf image” button.

Upload an image (.jpg, .jpeg, .png).

The app will display:

🧾 Predicted class (Early Blight / Late Blight / Healthy)

📊 Confidence score

📊 Model Details
Parameter	Description
Architecture	Custom CNN with Conv2D, MaxPooling, Dropout, Dense layers
Framework	TensorFlow / Keras
Optimizer	Adam
Loss Function	Categorical Crossentropy
Accuracy	96.8% (Validation)
Input Size	256 × 256 pixels
Augmentation	Rotation, zoom, shift, flip
🧾 Dataset

The model was trained on a Potato Leaf Disease Dataset, which includes:

Early Blight – leaves showing brown concentric rings

Late Blight – leaves with large dark lesions

Healthy – disease-free green leaves

Dataset sources:

PlantVillage Dataset (Kaggle)

🧩 Example Predictions
Uploaded Image	Prediction	Confidence

	Early Blight	97.6%

	Healthy	99.1%

	Late Blight	95.3%
🛠️ Technologies Used

Python 3.10+

TensorFlow / Keras

NumPy & Matplotlib

Streamlit

Pillow (PIL)

📈 Future Improvements

🧪 Add more disease categories

☁️ Deploy on cloud (Streamlit Cloud / Hugging Face Spaces)

🔄 Integrate REST API for mobile applications

👨‍💻 Author

Ashok Mishra
📘 Computer Science and Engineering, NIT Silchar
🌐 GitHub Profile

📜 License

This project is licensed under the MIT License – feel free to use and modify it for educational and research purposes.
