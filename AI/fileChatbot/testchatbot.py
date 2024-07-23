import os
from flask import Flask, request, abort, jsonify
import mysql.connector
import messagemodel1
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from pythainlp.util import thai_strftime
import random
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize

folderpath = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
@app.route('/diw', methods=['POST', 'GET'])
def Factories():
    state_tag_userid_path = ""
    if request.method == "POST":
        data_load = request.json
        messagetype = data_load["MessageType"]
        if messagetype == "Text":
            message = data_load["Msg"]
            message = message.replace(" ", "")
            if (message in messagemodel1.greating):
                    text_message = messagemodel1.greating_out  
                    return text_message
            elif (message in messagemodel1.Gas):
                    return messagemodel1.Gas[message]
            elif(message in messagemodel1.Law):
                    return messagemodel1.Law[message]
            else 











            
if __name__ == '__main__':
    #app.run(port=200, host="0.0.0.0", debug=True)   
    app.run(port=300, debug=False, host="0.0.0.0")
   