import os
from flask import Flask, request, abort, jsonify
import mysql.connector
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from pythainlp.util import thai_strftime
import random
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize 

folderpath = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
@app.route('/diw', methods=['POST', 'GET'])
def Factories():
    state_tag_userid_path = ""
    if request.method == "POST":
        data_load = request.json
        print(data_load)
        messagetype = data_load["MessageType"]
        if messagetype == "Text":
            message = data_load["Msg"]
            #message = tokenize_and_remove_stopwords(message)
            Ans = predict_first_model(message)
            print(Ans)
            return Ans
    elif (request.method == "GET"): #รีเควสGET 
        return "คำสั่งGETของChat Bot"

def tokenize_and_remove_stopwords(text): #กำหนดฟังก์ชั่น
    stop_word = set(thai_stopwords())
    tokens = word_tokenize (text,engine='newmm')
    return [token for token in tokens if token not in stop_word]

def predict_first_model(message):
    with open('vectorizer2.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open ('svm_model2.pkl', 'rb') as f:
        model = pickle.load(f)

    new_texts = [message]
    X_text = vectorizer.transform(new_texts)
    predictions = model.predict(X_text)

    for text, prediction in zip(new_texts, predictions):
        print(f"Text: {text} - Predicted tag: {prediction}")
    return predictions[0]

if __name__ == '__main__':
    #app.run(port=200, host="0.0.0.0", debug=True)   
    app.run(port=300, debug=False, host="0.0.0.0")

        