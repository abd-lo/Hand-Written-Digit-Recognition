# Handwritten Digit Recognition Web Application
### University Assignment Project

| Student ID | Name | GroupNum | Status |
|------------|------|------|--------|
| 342458 | ABD AL RAHMAN MOHAMMAD LOUBANI | 15 | |
| 339411 | Tamadur Nasser Ali | 15 | |
| 271462 | mhd nour mhd kussay kuzez | 15 | Didn't Respond/Participate |
| 307396 | Wael Yousef Hasan | 15 | Didn't Respond/Participate |


A simple web-based handwritten digit recognition system built using **Python, Flask, and TensorFlow**.

The application uses a neural network trained on the **MNIST handwritten digit dataset** to recognize digits from uploaded images. Users can upload an image containing a single handwritten digit (0-9), and the system predicts the digit with a confidence score.


<p align="center">
  <img width="400" alt="Handwritten Digit Recognition Demo" src="https://github.com/user-attachments/assets/529beef6-3d92-4f6b-86f5-4548b5306601" />
</p>

The model is trained using the **MNIST dataset**.

MNIST contains:

- 60,000 training images
- 10,000 testing images
- Images size: 28x28 pixels
- Classes: digits from 0 to 9

# Technologies Used

## Backend

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow (PIL)

## Frontend

- HTML
- CSS
- JavaScript

## Access Deployed Version:
https://handwritterdigittest.onrender.com/

## Running the Project Locally

Follow the steps below to run the project on your local machine.

### 1. Generate the Keras Model

Run the training script to train the model and generate the Keras model file:

```bash
python3 train.py
```

This step will create the trained model file that will be used by the application for predictions.

---

### 2. Start the Application

Run the backend application:

```bash
python3 app.py
```

The application will start and be ready to handle requests.

---

### 3. Open the Web Interface

Open the HTML interface located at:

```
templates/index.html
```

You can open it in your browser to interact with the application.
