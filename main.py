#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 2020

@author: sumon
"""

from __future__ import print_function
from flask import Flask, render_template
from flask import request, jsonify
from flask_ngrok import run_with_ngrok
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model

import os
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
run_with_ngrok(app)
app.debug = True
app._static_folder = os.path.abspath("templates/static/")

@app.route('/', methods=['GET'])
def index():
    title = 'Draw-Grapheme'
    return render_template('layouts/landing_page.html', title=title)

@app.route('/canvas', methods=['GET'])
def show():
    title = 'Draw-Grapheme'
    return render_template('layouts/index.html', title=title)

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
	alphabets = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
	grapheme_root= ['ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
       'ক', 'ক্ক', 'ক্ট', 'ক্ত', 'ক্ল', 'ক্ষ', 'ক্ষ্ণ', 'ক্ষ্ম', 'ক্স',
       'খ', 'গ', 'গ্ধ', 'গ্ন', 'গ্ব', 'গ্ম', 'গ্ল', 'ঘ', 'ঘ্ন', 'ঙ',
       'ঙ্ক', 'ঙ্ক্ত', 'ঙ্ক্ষ', 'ঙ্খ', 'ঙ্গ', 'ঙ্ঘ', 'চ', 'চ্চ', 'চ্ছ',
       'চ্ছ্ব', 'ছ', 'জ', 'জ্জ', 'জ্জ্ব', 'জ্ঞ', 'জ্ব', 'ঝ', 'ঞ', 'ঞ্চ',
       'ঞ্ছ', 'ঞ্জ', 'ট', 'ট্ট', 'ঠ', 'ড', 'ড্ড', 'ঢ', 'ণ', 'ণ্ট', 'ণ্ঠ',
       'ণ্ড', 'ণ্ণ', 'ত', 'ত্ত', 'ত্ত্ব', 'ত্থ', 'ত্ন', 'ত্ব', 'ত্ম', 'থ',
       'দ', 'দ্ঘ', 'দ্দ', 'দ্ধ', 'দ্ব', 'দ্ভ', 'দ্ম', 'ধ', 'ধ্ব', 'ন',
       'ন্জ', 'ন্ট', 'ন্ঠ', 'ন্ড', 'ন্ত', 'ন্ত্ব', 'ন্থ', 'ন্দ', 'ন্দ্ব',
       'ন্ধ', 'ন্ন', 'ন্ব', 'ন্ম', 'ন্স', 'প', 'প্ট', 'প্ত', 'প্ন', 'প্প',
       'প্ল', 'প্স', 'ফ', 'ফ্ট', 'ফ্ফ', 'ফ্ল', 'ব', 'ব্জ', 'ব্দ', 'ব্ধ',
       'ব্ব', 'ব্ল', 'ভ', 'ভ্ল', 'ম', 'ম্ন', 'ম্প', 'ম্ব', 'ম্ভ', 'ম্ম',
       'ম্ল', 'য', 'র', 'ল', 'ল্ক', 'ল্গ', 'ল্ট', 'ল্ড', 'ল্প', 'ল্ব',
       'ল্ম', 'ল্ল', 'শ', 'শ্চ', 'শ্ন', 'শ্ব', 'শ্ম', 'শ্ল', 'ষ', 'ষ্ক',
       'ষ্ট', 'ষ্ঠ', 'ষ্ণ', 'ষ্প', 'ষ্ফ', 'ষ্ম', 'স', 'স্ক', 'স্ট', 'স্ত',
       'স্থ', 'স্ন', 'স্প', 'স্ফ', 'স্ব', 'স্ম', 'স্ল', 'স্স', 'হ', 'হ্ন',
       'হ্ব', 'হ্ম', 'হ্ল', 'ৎ', 'ড়', 'ঢ়', 'য়']
	vowel_diacritic= ['', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
	consonant_diacritic = ['', 'ঁ', 'র্', 'র্য', '্য', '্র', '্র্য']
	data_url = request.form['canvas_data']
	offset = data_url.index(',')+1
	img_bytes = base64.b64decode(data_url[offset:])
	img = Image.open(BytesIO(img_bytes))
	img  = np.array(img)
	cv2.imwrite('picture.png',img)
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	classifier = model_from_json(loaded_model_json)
	# load weights into new model
	classifier.load_weights("model.h5")
	print("Loaded model from disk")
	img = cv2.imread("picture.png", cv2.IMREAD_GRAYSCALE)
	im_resized = cv2.resize(img, (64, 64))
	cv2.imwrite('trial2.png',im_resized)
	im = im_resized.reshape(64, 64, 1)
	im = im/255
	cv2.imwrite('seg.png',im_resized)
	im = im.reshape(1, 64, 64, 1)
	result = classifier.predict(np.array(im))
	print(result)
	preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
	}
	for i,p in enumerate(preds_dict):
		preds_dict[p] = np.argmax(result[i], axis=1)
	print(preds_dict)
	print(preds_dict['grapheme_root'][0])
	K.clear_session()
	params = { 
			   "consonant_diacritic": consonant_diacritic[preds_dict['consonant_diacritic'][0]],
			   "grapheme_root": grapheme_root[preds_dict['grapheme_root'][0]],
			   "vowel_diacritic" : vowel_diacritic[preds_dict['vowel_diacritic'][0]]
			   }
	return jsonify(params)

if __name__ == '__main__':
	app.run()
    # app.run(host='0.0.0.0', port=5000)
