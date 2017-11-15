import flask
from flask import Flask, Response, request, render_template, redirect, url_for, send_from_directory
from flaskext.mysql import MySQL
import flask_login as flask_login
import datetime
# for image uploading
# from werkzeug import secure_filename
import os, base64
from hash import CalcMD5


app.debug = True

mysql = MySQL()
app = Flask(__name__)
app.secret_key = 'super secret string'

#connect to mysql dataserver
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'PASSENGERSCREENING'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

# begin code used for login
login_manager = flask_login.LoginManager()
login_manager.init_app(app)

conn = mysql.connect()


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/results', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        imgfile = request.files['imageName']
        imgname = imgfile.filename
        cursor = conn.cursor()
        cursor.execute("INSERT INTO IMAGE (NAME) VALUES (%s)", (imgname))
        conn.commit()

        cursor = conn.cursor()
        cursor.execute("SELECT ID FROM IMAGE WHERE NAME = %s", imgname)
        pid = cursor.fetchone()[0]
        photo_url = str(pid)+ "." + (imgname.rsplit('.', 1)[1])
        imgfile.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_url))
        MD5 = CalcMD5(os.path.join(app.config['UPLOAD_FOLDER'], photo_url))
        cursor = conn.cursor()
        cursor.execute("UPDATE IMAGE SET MD5 = %f WHERE NAME = %s", (MD5,imgname))
        conn.commit()

        #reading img and run model get 17 Percentages
        P = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        cursor = conn.cursor()
        cursor.execute("UPDATE IMAGE SET P1 = %f,P2 = %f, P3 = %f,P4 = %f,P5 = %f,P6 = %f,P7 = %f,P8 = %f,P9 = %f,P10 = %f,P11 = %f,P12 = %f,P13 = %f,P14 = %f,P15 = %f,P16 = %f,P17 = %f WHERE NAME = %s", 
        				(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],P[9],P[10],P[11],P[12],P[13],P[14],P[15],P[16],P[17], imgname))
        conn.commit()

        return render_template('results.html', Perc=P)

    else:
    	P = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    	return render_template('results.html', Perc=P)


# end photo uploading code


#index page
@app.route("/", methods=['GET', 'POST'])
def index():
    	return render_template('index.html')


