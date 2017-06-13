import os
import subprocess
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROOT'] = os.path.dirname(os.path.abspath(__file__))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform(checkpoint, inputfile, outputfile):
    process = subprocess.Popen(['python','evaluate.py', '--checkpoint', str(checkpoint), '--in-path', str(inputfile), '--out-path', str(outputfile)], stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            painter = request.form.get('painter')

            #transform
            checkpoint = app.config['ROOT'] + '/checkpoint/rain-princess.ckpt'
            input_filename = app.config['UPLOAD_FOLDER'] + '/' + filename
            file_, ext_ = filename.split('.')
            outfile = file_ + '_afremov' + '.' + ext_
            output_filename = app.config['UPLOAD_FOLDER'] + '/' + outfile

            transform(checkpoint, input_filename, output_filename)
            #clean tmp

            #return redirect(url_for('upload_file',
            #                        filename=filename, painter=painter))
            return redirect(url_for('download',
                                    filename=outfile))
    return '''
    <!doctype html>
    <title>Photo Transformer</title>
    <h1>Photo Transformer</h1>
    <form method=post enctype=multipart/form-data>
        <select name="painter">
            <option value="afremov">Leonid Afremov</option>
            <option value="picasso">Pablo Picasso</option>
            <option value="picabia">Francis Picabia</option>
            <option value="munch" selected>Edvard Munch</option>
            <option value="hokusai">Hokusai</option>
            <option value="turner">J.M.W Turner</option>
        </select>
        <input type=file name=file>
        <input type=submit value=Transform>
    </form>
    '''

@app.route(app.config['UPLOAD_FOLDER']+'/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)

@app.route('/clean', methods=['GET', 'POST'])
def clean():
    process = subprocess.Popen(['rm', app.config['UPLOAD_FOLDER'] + '/*'], stdout=subprocess.PIPE)
    out = process.communicate()
    return out

if __name__ == "__main__":
    app.run(host='0.0.0.0')
