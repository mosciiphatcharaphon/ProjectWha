import os
from flask import Flask, request, abort, jsonify
import mysql.connector
import messagemodel2
import helper
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.util import thai_strftime
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize 

   

               