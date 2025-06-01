This project focuses on detecting potato diseases using a deep learning model built with a Convolutional Neural Network (CNN). It includes a model training pipeline and a backend API to serve the predictions.

Setup Instructions
Backend (Python & FastAPI)
Install Python packages:

bash
Copy
Edit
pip3 install -r training/requirements.txt  
pip3 install -r api/requirements.txt  
Run the FastAPI server:

bash
Copy
Edit
cd api  
uvicorn main:app --reload --host 0.0.0.0  
Model Training
Prepare your dataset (only potato images).

Run the training notebook:

Open training/potato-disease-training.ipynb in Jupyter Notebook.

Update the dataset path.

Run all cells to train the model.

Save the trained model in the models folder.

Summary
Deep Learning Model: Convolutional Neural Network (CNN)

Backend: FastAPI

