# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow 
import tflearn
import pickle
import json

#------------Static FAQ Bot Function----------------------
import string 
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)
with open('<Directory-name>/Q&A.txt','r', encoding='utf8', errors ='ignore') as fin:    
    raw = fin.read().lower()

#Sentence and word Tokenisation
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
#------------Static FAQ Bot Function----------------------


app = Flask(__name__)

with open('<directory-name>/ChatBot/intents.json') as file:
    data = json.load(file)

try:
    with open("<directory-name>/ChatBot/training_data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("<directory-name>/ChatBot/training_data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
#print(net)
net = tflearn.regression(net)


model = tflearn.DNN(net)

model.load('./ChatBot/model1.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

import pyodbc
server = '<server-name>'
database = '<db-name>'
username = '<user-name>'
    #password = 'mypassword'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes')
cursor = cnxn.cursor()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def chat():
    userText = request.args.get('msg')
    #print(userText)
    
#------------Static FAQ Bot Function----------------------
    lemmer = WordNetLemmatizer()
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    def LemNormalize(userText):
        return LemTokens(nltk.word_tokenize(userText.lower().translate(remove_punct_dict)))
    def response(user_response):
        robo_response=''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

#------------Static FAQ Bot Function----------------------
    while True:
            results = model.predict([bag_of_words(userText, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            if tag == 'user-satisfied':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you")
                elif results[results_index] >= 0.95:
                    return('You are welcome. Do you need any further help?')
            if tag == 'further-information':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you")
                elif results[results_index] >= 0.95:
                    return('I am glad to help you further. What other information do you need')
            if tag == 'user-leaving':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you")
                elif results[results_index] >= 0.95:
                    return("It's nice talking to you today. Bye! take care..")
            if tag == 'user-greeting':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you")
                elif results[results_index] >= 0.95:
                    return("Hi! I'm Henri! How can i help you today")
            if tag == 'contract-value':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you. Do you mean contract value?")
                elif results[results_index] >= 0.95:
                    contractid = int(''.join(filter(str.isdigit,userText)))
                    data=cursor.execute("<query>")
                    value = cursor.fetchone()
                    if value == None:
                        return ('There is no such contract available! Sorry')
                    elif value != None:
                        contract_value = ''.join(value)
                        total_value="The contract value is "+contract_value
                    #print(contract_value)
                        return(total_value)
            if tag == 'currency-name':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you. Do you mean currency type?")
                elif results[results_index] >= 0.95:
                    contractid = int(''.join(filter(str.isdigit,userText)))
                    data=cursor.execute("<query>")
                    value = cursor.fetchone()
                    if value == None:
                        return ('There is no such contract available! Sorry')
                    elif value != None:
                        currency_name = ''.join(value)
                        currency = "The contract currency is "+currency_name
                        return(currency)

            if tag == 'vendor-name':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you. Do you mean vendor name?")
                elif results[results_index] >= 0.95:
                    contractid = int(''.join(filter(str.isdigit,userText)))
                
                    data=cursor.execute("<query>")
                    value = cursor.fetchone()
                
                    if value == None:
                    #print('There is no such contract available! Sorry')
                        return('There is no such contract available! Sorry')
                    elif value != None:
                        vendor_name = ''.join(value)
                        vendor= "The vendor/supplier is "+vendor_name
                    #print(vendor_name) 
                        return(vendor)
            if tag == 'contract-type':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you. Do you mean contract type?")
                elif results[results_index] >= 0.95:
                    contractid = int(''.join(filter(str.isdigit,userText)))
                
                    data=cursor.execute("<query>")
                    value = cursor.fetchone()
                
                    if value == None:
                
                        return('There is no such contract available! Sorry')
                    elif value != None:
                        contract_type = ''.join(value)
                        type= "The contract category/commodity is "+contract_type
                    #print(contract_type) 
                        return(type)
            if tag == 'contract-language':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you. Do you mean contract language?")
                elif results[results_index] >= 0.95:
                    contractid = int(''.join(filter(str.isdigit,userText)))
                
                    data=cursor.execute("<query>")
                    value = cursor.fetchone()
                
                    if value == None:
                    #print('There is no such contract available! Sorry')
                        return('There is no such contract available! Sorry')
                    elif value != None:
                        contract_language = ''.join(value)
                        language= "The contract language is "+ contract_language
                    #print(contract_language) 
                        return(language)
            if tag == 'generic-qna':
                if results[results_index] <= 0.95:
                    return ("I am sorry! I didn't understand you")
                elif results[results_index] >= 0.95:
                    user_response = userText
                
                    return (response(user_response))
            
            


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8000)
