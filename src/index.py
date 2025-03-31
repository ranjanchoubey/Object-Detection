from flask import Flask, request, jsonify, send_file, make_response, render_template
from flask_cors import CORS
import io
import json
from inference import predict

# Create a Flask instance
app = Flask(__name__)
CORS(app)

# Define the home route
@app.route('/')
def home():
    # Render the HTML template
    return render_template('index.html')

# Define a route to handle image file uploads
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Predict and get annotated image and results
    annotated_image, detection_results = predict(image)

    # Create a bytes buffer for the image
    image_buffer = io.BytesIO(annotated_image)

    # Create a response that includes both the annotated image and detection results
    response = make_response(send_file(image_buffer, mimetype='image/jpeg', as_attachment=False))
    response.headers['Detection-Results'] = json.dumps(detection_results)

    return response

# Run the app
if __name__ == '__main__':
    # Start the Flask app on port 8000
    app.run(host='0.0.0.0', port=5001)
