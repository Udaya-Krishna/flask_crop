from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests
import json
from flask_mailman import Mail, EmailMessage
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load("crop_model.pkl")

# Load feature names (ensures correct input format)
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# OpenWeather API (Replace with your API key)
WEATHER_API_KEY = "233a169eca4c3d30d93928de0883ee9a"

# Email Configuration (Replace with your credentials)
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "ashwinnandacool@gmail.com"
app.config["MAIL_PASSWORD"] = "zotv xwkm fmzy woli"
app.config["MAIL_DEFAULT_SENDER"] = "ashwinnandacool@gmail.com"

mail = Mail(app)
mail.init_app(app)

def get_weather(city):
    """Fetch temperature and humidity for a given city using OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temperature, humidity
    else:
        return None, None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.json
        print("Received data:", data)  # Debug log
        
        # Extract all inputs directly from the request
        N = float(data.get("nitrogen"))
        P = float(data.get("phosphorus"))
        K = float(data.get("potassium"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        ph = float(data.get("ph"))
        rainfall = float(data.get("rainfall"))

        print("Processed values:", {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        })  # Debug log

        # Create DataFrame with correct feature names
        new_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
       
        # Get probability predictions for all classes
        probabilities = model.predict_proba(new_data)[0]
        
        # Get indices of top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        
        # Get the crop names and their probabilities
        top_3_crops = []
        for idx in top_3_indices:
            crop_name = model.classes_[idx]
            probability = probabilities[idx] * 100  # Convert to percentage
            top_3_crops.append({
                "crop": crop_name,
                "probability": round(probability, 2)
            })

        print("Top 3 predicted crops:", top_3_crops)  # Debug log
        return jsonify({"recommendations": top_3_crops})
    
    except Exception as e:
        print("Error occurred:", str(e))  # Debug log
        return jsonify({"error": str(e)}), 500

@app.route("/send_mail", methods=["POST"])
def send_mail():
    try:
        name = request.form["name"]
        recipient_email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]

        # Construct email body
        body = f"Name: {name}\n\nMessage:\n{message}"

        # Send email using Flask-Mailman
        email = EmailMessage(subject, body, to=[recipient_email])
        email.send()

        return "Email sent successfully!"
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
   app.run(debug=True)
