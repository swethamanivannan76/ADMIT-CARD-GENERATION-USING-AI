from flask import Flask, json, jsonify, render_template, request, redirect, send_file, url_for, Response
import os
import cv2
import numpy as np
from PIL import Image
import mysql.connector
import qrcode
from werkzeug.utils import secure_filename
from keras.models import load_model
from fpdf import FPDF
from keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from datetime import datetime
from keras.applications.vgg19 import preprocess_input
import pytesseract
import cv2
import numpy as np
import re
from twilio.rest import Client
import base64


app = Flask(__name__)

# Set your Twilio credentials
account_sid = "YOUR_SID"
auth_token = "YOUR_AUTH_TOKEN"
verify_sid = "YOUR_VERIFY_SID"

client = Client(account_sid, auth_token)

# Load the pretrained model for signature detection
signature_model = load_model("sign2.h5")

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Swetha\Desktop\tesseract\tesseract.exe'


UPLOAD_FOLDER = 'static/uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MySQL Database Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="admit_card_system"
)
cursor = db.cursor()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not loaded properly")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    image = cv2.resize(image, (256, 256))  # Resize to match the model's input shape
    image = image / 255.0
    return image

ref = {
    0: 0,
    1: 1
}
def prediction(path):
  img= load_img(path, target_size= (256,256) )

  i= img_to_array(img)
  im= preprocess_input(i)
  img=np.expand_dims(im , axis= 0)
  pred = np.argmax(signature_model.predict(img))
    
  return ref[pred]


def verify_signature(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = signature_model.predict(np.array([preprocessed_image]))
    return prediction[0][0] >= 0.5

def verify_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Admit Card', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_admit_card(user):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    pdf.rect(10, 20, 190, 150)

    pdf.ln(15)
    pdf.set_font('Arial', 'B', 16)  # Set font to bold and size 16
    pdf.cell(0, 10, 'Student ID', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)  # Reset font to regular and size 12
    pdf.ln(10)
    pdf.cell(80, 10, f'Name: {user[1]}', 0, 1,'L')  
    pdf.cell(40, 10, f'DOB: {user[2]}', 0, 1,'L')   
    pdf.cell(40, 10, f'Address: {user[3]}', 0, 1,'L')  
    pdf.cell(40, 10, f'Gender: {user[4]}', 0, 1,'L')  
    pdf.cell(40, 10, f'Email: {user[5]}', 0, 1,'L')  
    pdf.cell(40, 10, f'Phone: {user[6]}', 0, 1,'L') 

    # Load and add the person's photo to the PDF
    person_photo = Image.open(user[7])
    person_photo = person_photo.resize((60, 60))  
    pdf.image(user[7], 150, 50, 30, 30) 
    signature = Image.open(user[8])
    signature = signature.resize((50, 12))  
    pdf.image(user[8], 145, 140, 50, 12)  


    pdf.output('static/admit_card.pdf')

def generate_qr_data(user_data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    # Generate the QR code content with user data
    qr_data = f"Name: {user_data['name']}\nDOB: {user_data['dob']}\nAddress: {user_data['address']}\nGender: {user_data['gender']}\nEmail: {user_data['email']}\nPhone: {user_data['phone']}"

    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Encode the user's photo as base64 and include it in the QR code content
    encoded_image = user_data['encoded_image']
    qr_img.paste(encoded_image, (100, 100))  # Adjust the position as needed

    qr_img_path = 'static/temp_qr.png'
    qr_img.save(qr_img_path)

    return qr_img_path



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/generate_admit_card', methods=['GET', 'POST'])
def generate_admit_card_route():
    error_message = None
    qr_code_data = None
    if request.method == 'POST':
        input_number = request.form['phone_number']
        phone_number = '+91' + input_number
        user_otp = request.form['otp']  # User-entered OTP

        # Verify the user's OTP using Twilio Verify service
        verification_check = client.verify.v2.services(verify_sid) \
            .verification_checks \
            .create(to=phone_number, code=user_otp)

        if verification_check.status == 'approved':
            cursor.execute("SELECT * FROM users WHERE phone = %s", (input_number,))
            user = cursor.fetchone()

            if user:
                 # Convert datetime.date to a formatted string
                formatted_dob = user[2].strftime('%Y-%m-%d')  # Assuming DOB is in the third index
                
                # Create a new tuple with the modified DOB
                modified_user = tuple(user[:2] + (formatted_dob,) + user[3:])
                
                # Generate QR code data containing user information
                qr_code_data = "\n".join(map(str, modified_user[1:]))
                generate_admit_card(user)
                
                return render_template('admit_card.html', user=modified_user, qr_code_data=qr_code_data)
            else:
                 error_message = "User not found"
        else:
            error_message = "Incorrect OTP"

    return render_template('generate_admit_card.html', error_message=error_message)

@app.route('/send_otp', methods=['POST'])
def send_otp():
    input_number = request.json.get('phone_number')
    phone_number = '+91' + input_number

    verification = client.verify.v2.services(verify_sid) \
        .verifications \
        .create(to=phone_number, channel="sms")

    if verification.status == 'pending':
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})




@app.route('/generate_qr', methods=['POST'])
def generate_qr():
    user_data = request.get_json()

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    qr_data = f"Name: {user_data['name']}\n" \
               f"DOB: {user_data['dob']}\n" \
               f"Address: {user_data['address']}\n" \
               f"Gender: {user_data['gender']}\n" \
               f"Email: {user_data['email']}\n" \
               f"Phone: {user_data['phone']}"

    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    img_path = 'temp_qr.png'
    img.save(img_path)

    return send_file(img_path, mimetype='image/png')


@app.route('/download_admit_card', methods=['GET'])
def download_admit_card():
    try:
        return send_file('static/admit_card.pdf', as_attachment=True)
    except Exception as e:
        return f"Error while downloading: {e}"


@app.route('/enter_details', methods=['GET', 'POST'])
def enter_details():
    error_messages = {
        'photo': None,
        'signature': None,
        'aadhar_image': None,
        'aadhar_mismatch': None,
        'name': None,
        'both': None
    }
    error_message= None
    
    if request.method == 'POST':
        # Extract form data
        name = request.form['name']
        dob = request.form['dob']
        address = request.form['address']
        gender = request.form['gender']
        email = request.form['email']
        phone = request.form['phone']
        entered_aadhar_number = request.form['aadhar_number']


        # Validate and insert data into the database
        try:
            # Process and save images
            photo = request.files['photo']
            signature = request.files['signature']
            aadhar_image = request.files['aadhar_image']

            if not photo:
                error_messages['photo'] = "Please upload a photo."
            if not signature:
                error_messages['signature'] = "Please upload a signature."
            if not aadhar_image:
                error_messages['aadhar_image'] = "Please upload an Aadhar image."

            
            if photo and signature and aadhar_image:
                photo_filename = secure_filename(photo.filename)
                signature_filename = secure_filename(signature.filename)
                aadhar_image_filename = secure_filename(aadhar_image.filename)
                photo_path = os.path.join(UPLOAD_FOLDER, photo_filename)
                signature_path = os.path.join(UPLOAD_FOLDER, signature_filename)
                aadhar_image_path = os.path.join(UPLOAD_FOLDER, aadhar_image_filename)


                # Save images to upload folder
                photo.save(photo_path)
                signature.save(signature_path)
                #aadhar_image.save(aadhar_image_path)
            

                # Verify images
                is_photo = verify_face(photo_path)
                is_signature = prediction(signature_path)

                       # OCR on the Aadhar image
                aadhar_image_data = aadhar_image.read()
                aadhar_image_array = np.frombuffer(aadhar_image_data, np.uint8)
                aadhar_cv2_image = cv2.imdecode(aadhar_image_array, cv2.IMREAD_COLOR)
                extracted_aadhar_text = pytesseract.image_to_string(aadhar_cv2_image, lang='eng')


                if not is_photo and not is_signature:
                    # Both verification failed, try the opposite images
                    is_photo_with_signature = verify_face(signature_path)
                    is_signature_with_photo = prediction(photo_path)

                    if is_photo_with_signature and is_signature_with_photo:
                        user_photo_path = signature_path
                        user_signature_path = photo_path
                    elif is_signature_with_photo:
                        user_photo_path = photo_path
                        user_signature_path = signature_path
                    elif not is_signature_with_photo:    
                        error_messages['signature']= "Signature is invalid"
                    else:
                        error_messages['both']= "Both images are invalid. Please upload a valid photo and signature."
                elif not is_photo:
                    is_signature = prediction(photo_path)
                    if is_signature:
                        user_photo_path = signature_path
                        user_signature_path = photo_path
                    else:
                        error_messages['photo']= "Photo is invalid."
                elif not is_signature:
                    is_photo = verify_face(signature_path)
                    if is_photo:
                        user_photo_path = photo_path
                        user_signature_path = signature_path
                    else:
                        error_messages['signature']= "Signature is invalid."
                else:
                    # Both images are valid
                    user_photo_path = photo_path
                    user_signature_path = signature_path

                    # Extract Aadhar number from the extracted text
                aadhar_pattern = r"\d{4}\s?\d{4}\s?\d{4}"
                aadhar_match = re.search(aadhar_pattern, extracted_aadhar_text)
                
                if aadhar_match:
                    extracted_aadhar_number = aadhar_match.group()
                    extracted_aadhar_number = extracted_aadhar_number.replace(" ", "")  # Remove spaces
                else:
                    extracted_aadhar_number = None

                # Check if the entered name is present in the extracted text (case-insensitive)
                if name.lower() not in extracted_aadhar_text.lower():
                    error_messages['name'] = "Name not found in Aadhar."

                # Check if the Aadhar number and entered Aadhar number match
                if extracted_aadhar_number and entered_aadhar_number != extracted_aadhar_number:
                    error_messages['aadhar_mismatch'] = "Aadhar number and entered Aadhar number do not match."

                is_photo_int = int(is_photo)
                if not any(error_messages.values()):
                    # Insert valid data into the database
                    cursor.execute("INSERT INTO users (name, dob, address, gender, email, phone, aadhar_number, photo_path, signature_path, is_photo, is_signature) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (name, dob, address, gender, email, phone, entered_aadhar_number, user_photo_path, user_signature_path, is_photo_int, is_signature))
                    db.commit()

                    return redirect(url_for('index'))
        except Exception as e:
            db.rollback()
            return f"Error: {e}"

    return render_template('enter_details.html', error_messages=error_messages,error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
