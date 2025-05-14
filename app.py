import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from utils.augmentor import process_image
import zipfile
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUGMENTED_FOLDER'] = 'static/augmented'
app.config['ZIP_FOLDER'] = 'zip_output'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUGMENTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['ZIP_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    mode = request.form.get('mode')
    augmentations = request.form.getlist('augmentations')

    # Clear previous augmented images
    for f in os.listdir(app.config['AUGMENTED_FOLDER']):
        os.remove(os.path.join(app.config['AUGMENTED_FOLDER'], f))

    # Process the image
    output_images = process_image(filepath, mode, augmentations, app.config['AUGMENTED_FOLDER'])

    # Create ZIP file
    zip_id = str(uuid.uuid4())
    zip_path = os.path.join(app.config['ZIP_FOLDER'], f"{zip_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_path in output_images:
            zipf.write(img_path, os.path.basename(img_path))

    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT if available
    app.run(debug=True, host='0.0.0.0', port=port)
