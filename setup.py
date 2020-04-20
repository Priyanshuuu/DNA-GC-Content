# Flask Packages
from flask import Flask,render_template,request,url_for,Response
from flask_bootstrap import Bootstrap 

import io
import random
from flask import Flask, Response, request
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io
import random
from werkzeug import secure_filename

import datetime
import time

# ML Packages
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn import model_selection

# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import render_template, url_for, flash, redirect, request

import os
from whoosh.analysis import StemmingAnalyzer

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo




app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

whoosh = os.path.join('static')
app.config['WHOOSH_BASE'] = whoosh
# set the global analyzer, defaults to StemmingAnalyzer.
app.config['WHOOSH_ANALYZER'] = StemmingAnalyzer()

PEOPLE_FOLDER = os.path.join('static', 'images')

db = SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)


@app.route('/')
@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('layout.html')


@app.route("/about", methods=['GET', 'POST'])
def about():
    return render_template('about.html', title='About')


@app.route("/login", methods=['GET', 'POST'])
def login():
	class LoginForm(FlaskForm):
		email = StringField('Email',validators=[DataRequired(), Email()])
		password = PasswordField('Password', validators=[DataRequired()])
		remember = BooleanField('Remember Me')
		submit = SubmitField('Login')
	form = LoginForm()
	credential_id = ['abhitiwari299@gmail.com', 'pandeyprince25@gmail.com', 'gaurpratima02@gmail.com'] 
	if form.validate_on_submit():
	    if (form.email.data in credential_id) and (form.password.data == 'gizmowits'):
	        flash('You have been logged in!', 'success')
	        return redirect(url_for('index'))
	    else:
	        flash('Login Unsuccessful. Please check username and password', 'danger')
	return render_template('login.html', title='Login', form=form)


@app.route("/register", methods=['GET', 'POST'])
def register():
	class RegistrationForm(FlaskForm):
		username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
		email = StringField('Email', validators=[DataRequired(), Email()])
		password = PasswordField('Password', validators=[DataRequired()])
		confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
		submit = SubmitField('Sign Up')
	form = RegistrationForm()
	if form.validate_on_submit():
	    flash('Account created for {form.username.data}!', 'success')
	    return redirect(url_for('home'))
	return render_template('register.html', title='Register', form=form)




@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template('index.html')

# Route for our Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB',filename))
		fullfile = os.path.join('static/uploadsDB',filename)

		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		# EDA function
		data = pd.read_csv(os.path.join('static/uploadsDB',filename)).T
		data.columns = list(data.iloc[0])
		data.drop(['0'],inplace=True)
		df = data
		# Standardize the data to have a mean of ~0 and a variance of 1
		X_std = StandardScaler().fit_transform(df)# Create a PCA instance: pca
		pca = PCA(n_components=20)
		principalComponents = pca.fit_transform(X_std)# Plot the explained variances
		features = range(pca.n_components_)
		plt.close('all')
		plt.bar(features, pca.explained_variance_ratio_, color='black')
		plt.xlabel('PCA features')
		plt.ylabel('variance %')
		plt.xticks(features)# Save components to a DataFrame
		plt.savefig('static/images/data1.png')
		full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'data1.png')
		plt.close('all')
		
		#-----Section-1-----

		PCA_components = pd.DataFrame(principalComponents)
		plt.scatter(PCA_components[18], PCA_components[19], alpha=.1, color='black')
		plt.xlabel('PCA 1')
		plt.ylabel('PCA 2')
		plt.savefig('static/images/data2.png')
		full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'data2.png')
		plt.close('all')

		#-----Section-2-----

		plt.scatter(PCA_components[0], PCA_components[19], alpha=.1, color='black')
		plt.xlabel('PCA 1')
		plt.ylabel('PCA 2')
		plt.savefig('static/images/data3.png')
		full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'data3.png')
		plt.close('all')

		#-----Section-3-----

		from sklearn.cluster import KMeans
		ks = range(1, 10)
		inertias = []
		for k in ks:
		    # Create a KMeans instance with k clusters: model
		    model = KMeans(n_clusters=k)
		    
		    # Fit model to samples
		    model.fit(PCA_components.iloc[:,:3])
		    
		    # Append the inertia to the list of inertias
		    inertias.append(model.inertia_)
		    
		plt.plot(ks, inertias, '-o', color='black')
		plt.xlabel('number of clusters, k')
		plt.ylabel('inertia')
		plt.xticks(ks)
		plt.savefig('static/images/data4.png')
		full_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'data4.png')
		plt.close('all')

		#-----Section-4-----

		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=4)
		kmeans.fit(PCA_components.iloc[:,:4])
		y_kmeans = kmeans.predict(PCA_components.iloc[:,:4]) 
		 
		plt.scatter(PCA_components.iloc[:, 0], PCA_components.iloc[:, 1], c=y_kmeans, s=30, cmap='viridis')

		centers = kmeans.cluster_centers_
		plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
		plt.savefig('static/images/data5.png')
		full_filename5 = os.path.join(app.config['UPLOAD_FOLDER'], 'data5.png')
		plt.close('all')

		#-----Section-5-----

		# from mpl_toolkits.mplot3d import Axes3D
		# kmeans = KMeans(n_clusters=4)
		# fig = plt.figure(1, figsize=(4, 3))
		# ax = Axes3D(fig, rect=[1, 1, 1.95, 1.90], elev=30, azim=50)
		# kmeans.fit(PCA_components.iloc[:,:4])
		# labels = kmeans.labels_
		# x = PCA_components.iloc[:,0]
		# y = PCA_components.iloc[:,1]
		# z = PCA_components.iloc[:,2]
		# ax.scatter(x, y, z,c=labels.astype(np.float), edgecolor='k')
		# ax.set_xlabel('X-axis')
		# ax.set_ylabel('Y-axis')
		# ax.set_zlabel('Z-axis')
		# ax.set_title('4 Cluster Graph')
		# ax.dist = 10
		#-----Section-6-----
   	
	return render_template('report.html', PCA = full_filename1,
		SVC = full_filename2,
		DVC = full_filename3,
		IVML = full_filename4,
		TWOD = full_filename5 )

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
	import webbrowser
	# webbrowser.open("http://127.0.0.1:5000/")
	app.run(debug=True)
