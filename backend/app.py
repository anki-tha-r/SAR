import os
import cv2
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from predict import predict_image
from flask_cors import CORS
import segmentation_models_pytorch as smp
from land.land_detection import predict_mask_on_image 
from land.land_app import land_bp  
from building.predict import predict_image as predict_building_image



app = Flask(__name__)
app.register_blueprint(land_bp)

CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16MB

app.config['UPLOAD_FOLDER_LAND'] = 'land/static/uploads'
app.config['RESULTS_FOLDER_LAND'] = 'land/static/outputs'

app.config['UPLOAD_FOLDER_BUILDING'] = 'building/uploads'
app.config['RESULTS_FOLDER_BUILDING'] = 'building/outputs'


# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

os.makedirs(app.config['UPLOAD_FOLDER_LAND'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER_LAND'], exist_ok=True)

os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER_BUILDING']), exist_ok=True)
os.makedirs(os.path.join(app.root_path, app.config['RESULTS_FOLDER_BUILDING']), exist_ok=True)



@app.route('/road_output', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Predict and save output mask
            output_mask = predict_image(input_path)
            output_filename = 'mask_' + filename
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            import cv2
            cv2.imwrite(output_path, output_mask)

            input_image = url_for('static', filename='uploads/' + filename)
            output_image = url_for('static', filename='outputs/' + output_filename)

            # Return the paths as JSON for dynamic update
            return jsonify({'input_image': input_image, 'output_image': output_image})

    return render_template('index.html')

@app.route('/building_output', methods=['GET', 'POST'])
def building_output():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            # full paths under backend/building/uploads and backend/building/outputs
            upload_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER_BUILDING'])
            out_dir = os.path.join(app.root_path, app.config['RESULTS_FOLDER_BUILDING'])
            input_path = os.path.join(upload_dir, filename)
            file.save(input_path)

            # run building predictor (returns BGR numpy image ready for cv2.imwrite)
            output_mask = predict_building_image(input_path)

            output_filename = 'mask_' + filename
            output_path = os.path.join(out_dir, output_filename)
            cv2.imwrite(output_path, output_mask)

            # return URLs that will be served by building_static route below
            input_url = url_for('building_static', filename=f'uploads/{filename}')
            output_url = url_for('building_static', filename=f'outputs/{output_filename}')
            return jsonify({'input_image': input_url, 'output_image': output_url})
        
    return render_template('building_index.html')

# Serve files from backend/building via /building_static/<path:filename>
@app.route('/building_static/<path:filename>')
def building_static(filename):
    building_dir = os.path.join(app.root_path, 'building')
    return send_from_directory(building_dir, filename)



# @app.route('/land_output', methods=['GET', 'POST'])
# def land_output():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = secure_filename(file.filename)
#             input_path = os.path.join(app.config['UPLOAD_FOLDER_LAND'], filename)
#             file.save(input_path)

#             # Run land-water detection
#             output_filename = 'mask_' + filename
#             output_path = os.path.join(app.config['OUTPUT_FOLDER_LAND'], output_filename)

#             output_mask = land_water_detection(input_path, output_path)

#             input_image = url_for('land/static', filename='uploads/' + filename)
#             output_image = url_for('land/static', filename='outputs/' + output_filename)

#             return jsonify({'input_image': input_image, 'output_image': output_image})

#     return render_template('land_index.html')



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8001, debug=True)
