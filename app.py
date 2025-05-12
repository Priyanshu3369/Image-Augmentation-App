from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils.augmentor import process_image
import zipfile
import uuid
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUGMENTED_FOLDER'] = 'static/augmented'
app.config['ZIP_FOLDER'] = 'zip_output'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUGMENTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['ZIP_FOLDER'], exist_ok=True)

# Global flag for tracking background task
processing_done = False
zip_result_path = ""

def background_task(files, mode, augmentations, zip_id):
    global processing_done, zip_result_path

    # Clear old
    for f in os.listdir(app.config['AUGMENTED_FOLDER']):
        os.remove(os.path.join(app.config['AUGMENTED_FOLDER'], f))

    # Process all images
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        process_image(filepath, mode, augmentations, app.config['AUGMENTED_FOLDER'], count=500)

    # Zip all images
    zip_path = os.path.join(app.config['ZIP_FOLDER'], f"{zip_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img in os.listdir(app.config['AUGMENTED_FOLDER']):
            zipf.write(os.path.join(app.config['AUGMENTED_FOLDER'], img), img)

    zip_result_path = zip_path
    processing_done = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global processing_done
    processing_done = False

    mode = request.form.get('mode')
    augmentations = request.form.getlist('augmentations')
    files = request.files.getlist('image')

    zip_id = str(uuid.uuid4())
    threading.Thread(target=background_task, args=(files, mode, augmentations, zip_id)).start()

    return redirect(url_for('progress'))

@app.route('/progress')
def progress():
    if processing_done:
        return redirect(url_for('download'))
    return "Processing... please wait and refresh every 10 seconds."

@app.route('/download')
def download():
    return send_file(zip_result_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
