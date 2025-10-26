# """
# Flask web UI to upload an image and view prediction.
# Run: python app.py
# """
# import os
# from flask import Blueprint,Flask, render_template, request, redirect, url_for, send_from_directory
# from werkzeug.utils import secure_filename
# import cv2
# from land.land_detection import predict_mask_on_image
# import segmentation_models_pytorch as smp

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
# CHECKPOINT = os.path.join(BASE_DIR, 'land/checkpoints', 'best_model.pth')  # adjust if needed
# ENCODER = 'resnet50'

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ALLOWED_EXTS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

# # --- Define Blueprint ---
# land_bp = Blueprint(
#     'land_bp',
#     __name__,
#     template_folder='templates',
#     static_folder='static'
# )


# @land_bp.route('/land_output', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return redirect(request.url)
#         file = request.files['image']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(save_path)

#             # get preprocessing fn for encoder
#             preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
#             result_mask, _ = predict_mask_on_image(save_path, checkpoint=CHECKPOINT, encoder=ENCODER, preprocess_fn=preprocess_fn)
#             result_name = f"{os.path.splitext(filename)[0]}_pred.png"
#             result_path = os.path.join(app.config['RESULTS_FOLDER'], result_name)
#             # save BGR
#             cv2.imwrite(result_path, result_mask[:, :, ::-1])

#             return render_template('land_index.html', input_image=url_for('uploaded_file', filename=filename),
#                                    result_image=url_for('result_file', filename=result_name))

#     return render_template('land_index.html')


# @land_bp.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# @land_bp.route('/results/<filename>')
# def result_file(filename):
#     return send_from_directory(app.config['RESULTS_FOLDER'], filename)

from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, current_app
from werkzeug.utils import secure_filename
import cv2
import os
import segmentation_models_pytorch as smp
from land.land_detection import predict_mask_on_image

ENCODER = 'resnet50'
CHECKPOINT = os.path.join('land/checkpoints', 'best_model.pth')  # adjust if needed

ALLOWED_EXTS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

land_bp = Blueprint(
    'land_bp',
    __name__,
    template_folder='templates',
    static_folder='static'
)

@land_bp.route('/land_output', methods=['GET', 'POST'])
def land_output():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(current_app.config['UPLOAD_FOLDER_LAND'], filename)
            file.save(save_path)

            preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
            result_mask, _ = predict_mask_on_image(
                save_path,
                checkpoint=CHECKPOINT,
                encoder=ENCODER,
                preprocess_fn=preprocess_fn
            )

            result_name = f"{os.path.splitext(filename)[0]}_pred.png"
            result_path = os.path.join(current_app.config['RESULTS_FOLDER_LAND'], result_name)
            cv2.imwrite(result_path, result_mask[:, :, ::-1])

            return render_template(
                'land_index.html',
                input_image=url_for('land_bp.uploaded_file', filename=filename),
                result_image=url_for('land_bp.result_file', filename=result_name)
            )

    return render_template('land_index.html')


@land_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER_LAND'], filename)


@land_bp.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(current_app.config['RESULTS_FOLDER_LAND'], filename)
