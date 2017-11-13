import flask
from flask import Flask, Response, request, render_template, redirect, url_for, send_from_directory
from flaskext.mysql import MySQL
import flask_login as flask_login
import datetime
# for image uploading
# from werkzeug import secure_filename
import os, base64
from hash import CalcMD5

mysql = MySQL()
app = Flask(__name__)
app.secret_key = 'super secret string'

#connect to mysql dataserver
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'PASSENGERSCREENING'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)


# begin photo uploading code
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/results', methods=['POST'])
def process_image():
    if request.method == 'POST':
        imgfile = request.files['imageName']
        imgname = imgfile.filename
        MD5 = CalcMD5(imgfile)
        if imgfile and allowed_file(imgname):

            cursor = conn.cursor()
            cursor.execute("INSERT INTO IMAGE (NAME, MD5) VALUES (%s, %s, %s)", (imgname, MD5))
            conn.commit()

            cursor = conn.cursor()
            cursor.execute("SELECT ID FROM IMAGE WHERE NAME = %s", imgname)
            pid = cursor.fetchone()[0]
            photo_url = str(pid)+ "." + (imgname.rsplit('.', 1)[1])
            imgfile.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_url))

            #reading img and run model get 17 Percentages

            cursor = conn.cursor()
            cursor.execute("UPDATE IMAGE SET P1 = %f WHERE NAME = %s", (plist, imgname))
            conn.commit()

            return render_template('results.html', Perc=percent)


# end photo uploading code


#index page
@app.route("/", methods=['GET', 'POST'])
def index():
    	return render_template('index.html')