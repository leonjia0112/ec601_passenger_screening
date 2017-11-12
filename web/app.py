import flask
from flask import Flask, Response, request, render_template, redirect, url_for, send_from_directory
from flaskext.mysql import MySQL
import flask_login as flask_login
import datetime
# for image uploading
# from werkzeug import secure_filename
import os, base64

mysql = MySQL()
app = Flask(__name__)
app.secret_key = 'super secret string'

#connect to mysql dataserver
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'PassengerScreening'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)


# # begin photo uploading code
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# UPLOAD_FOLDER = './uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/uploads/<path:filename>')
# def uploaded_file(filename):
#     print(filename)
#     return  send_from_directory("uploads/", filename, as_attachment = True)

# @app.route('/alrename/<albumID>', methods=[ 'GET','POST'])
# @flask_login.login_required
# def rename_album(albumID):
#     if request.method == "POST":
#         try:
#             newname = request.form.get('newname')
#         except:
#             print("input is blank")  # this prints to shell, end USER will not see this (all print statements go to shell)
#             return render_template('rename.html', message = "Album name can't be blank!")
#         cursor = conn.cursor()
#         cursor.execute("UPDATE ALBUM SET NAME = %s WHERE AID = %s", (newname, albumID))
#         conn.commit()
#         return redirect(url_for('album_detail', albumID=albumID))
#     else:
#         return render_template('rename.html',aid=albumID)



# @app.route('/upload/<aid>', methods=['GET', 'POST'])
# @flask_login.login_required
# def upload_file(aid):
#     if request.method == 'POST':
#         imgfile = request.files['photo']
#         imgname = imgfile.filename
#         tags = request.form.get("tags")
#         print(tags)
#         tags = tags.split('#')
#         if imgfile and allowed_file(imgname):
#             caption = request.form.get('caption')
#             photo_url = caption
#             cursor = conn.cursor()
#             cursor.execute("INSERT INTO PHOTO (PHOTOURL, AID, CAPTION) VALUES (%s, %s, %s)", (photo_url, aid, caption))
#             conn.commit()
#             cursor = conn.cursor()
#             cursor.execute("SELECT PID FROM PHOTO WHERE PHOTOURL = %s", photo_url)
#             pid = cursor.fetchone()[0]
#             print(pid)
#             photo_url = str(pid)+ "." + (imgname.rsplit('.', 1)[1])
#             imgfile.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_url))
#             cursor = conn.cursor()
#             cursor.execute("UPDATE PHOTO SET PHOTOURL = %s WHERE PHOTOURL = %s", (photo_url, caption))
#             conn.commit()
#             print(tags)
#             for i in range(1, len(tags)):
#                 cursor = conn.cursor()
#                 cursor.execute("SELECT HASHTAG FROM TAG WHERE HASHTAG = %s", tags[i])
#                 res = cursor.fetchall()
#                 if len(res) == 0 and tags[i] != " ":
#                     cursor = conn.cursor()
#                     cursor.execute("INSERT INTO TAG (HASHTAG) VALUES (%s)", (tags[i]))
#                     conn.commit()
#                 cursor = conn.cursor()
#                 cursor.execute("SELECT HASHTAG FROM ASSOCIATE WHERE HASHTAG = %s AND PID = %s", (tags[i], pid))
#                 exi = cursor.fetchall()
#                 if len(exi) == 0:
#                     cursor.execute("INSERT INTO ASSOCIATE (HASHTAG, PID) VALUES (%s,%s)", (tags[i], pid))
#                     conn.commit()
#             return redirect( url_for('album_detail', albumID = aid))

#     # The method is GET so we return a  HTML form to upload the a photo.
#     else:
#         cursor = conn.cursor()
#         cursor.execute("SELECT HASHTAG FROM ASSOCIATE GROUP BY HASHTAG ORDER BY COUNT(PID) DESC LIMIT 10")
#         tags = cursor.fetchall()
#         return render_template('upload.html', aid = aid, tags =tags )

# # end photo uploading code


#index page
@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')