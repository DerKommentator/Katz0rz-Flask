import imghdr
import tensorflow as tf
from flask import Flask, request, render_template, abort
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil

#try:
#    shutil.rmtree('uploaded/image/')
#except:
#    pass

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

model = tf.keras.models.load_model('Cats-Dogs-64x3-CNN.model')


app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "tobi": generate_password_hash("HwRq/"),
    "gast": generate_password_hash("kekw1234!")
}

CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_FOLDER'] = 'static/uploaded/image'


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


def finds():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_dir = 'static/uploaded'

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(50, 50),
        color_mode="grayscale",
        shuffle=False,
        class_mode='binary',
        batch_size=1
    )

    prediction = model.predict(test_generator)
    print(prediction)
    return np.round(prediction)[0][0]


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Deletes every file in the upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('%s konnte nicht gelöscht werden. Exception: %s' % (file_path, e))

        labels = ['Dog', 'Cat']
        f = request.files['file']

        # File Validation
        filename = secure_filename(f.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(f.stream):
                abort(400)

            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            val = finds()
            print(val)
            data = {'score': str(val), 'pred_label': labels[int(val)]}
            return data


if __name__ == '__main__':
    app.run(host='0.0.0.0')
