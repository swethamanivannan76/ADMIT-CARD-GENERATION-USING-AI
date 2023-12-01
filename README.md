# Admit Card System
This Flask application serves as an Admit Card generation system, allowing users to enter their details, verify their identity through photo and signature, and generate a downloadable Admit Card in PDF format.

## Features
OTP Verification: Utilizes Twilio Verify service to verify user identity through phone number OTP.

Face and Signature Verification: Validates user-submitted photos and signatures for identity verification.

QR Code Generation: Generates QR codes containing user information for easy access and sharing.

## Installation
Clone the repository:

```bash
git clone https://github.com/your-username/admit-card-system.git
```
Install the required Python packages:

```bash
pip install -r requirements.txt
```
### Set up MySQL database:

Create a MySQL database named admit_card_system.
Update the MySQL connection details in the app.py file.

## Run the application:

```bash
python app.py
```

The application will be accessible at http://localhost:5000/.

## Usage
Access the application through the provided URL.

Navigate to the respective sections for entering details, generating Admit Cards, and more.

Follow the on-screen instructions for OTP verification and document submission.

Download the generated Admit Card in PDF format.

## Dependencies
Flask
OpenCV
NumPy
Pillow
MySQL Connector
PyTesseract
Twilio
Keras
TensorFlow
## Contributors
Harishh A
Swetha M, Dinesh L &Sridevi T


Feel free to contribute to the project by submitting issues or pull requests.
