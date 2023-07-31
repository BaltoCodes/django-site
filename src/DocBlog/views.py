from datetime import datetime

from django.shortcuts import render
from binance import Client

import requests
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt



def index(request):
    
    return render(request, "DocBlog/index.html", context={"date": datetime.today()})

def interactive_graph(request):
   data=generation_data('BTCUSDT', '1h')
  
   return render(request, "DocBlog/interactive_graph.html", context={"data": data})

def world_is_yours(request):
    

    return render(request, 'DocBlog/world.html')


def new_world(request):
   return render(request, "DocBlog/new_earth.html")

def human(request):
   return render(request,"DocBlog/human.html" )

def accueil(request):

    import requests
    # Récupérer les données du prix du Bitcoin (utilisation d'une API publique)
    response = requests.get('https://api.coindesk.com/v1/bpi/currentprice/BTC.json')
    data = response.json()
    bitcoin_price = data['bpi']['USD']['rate']

    # Passer les données au template pour l'affichage
    context = {
        'title': 'My titre',
        'bitcoin_price': bitcoin_price,
    }
    
    return render(request, "DocBlog/accueil.html", context=context)


import matplotlib.pyplot as plt
import io
import urllib, base64



def generation_data(coin,interval):
    from datetime import timezone
    df={}
    #df1=pd.DataFrame(columns=pairList)
    #for pair in pairList:
        # print(pair)
    binance = Client('y4pd3mx4kPQ5drHGA7xtv7xuUCobXcBJSJJ54zV5oZmAz4RXgEXwAJ9uEmzwarD2','qMr0iTqa1byVDs3xHvq0BKGO29msysaFuKYp9zKkGDa4SThE97XTbufKXyEo9J24')

    dfList = pd.DataFrame(columns= ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time'])
    candle = binance.get_klines(symbol = coin, interval =interval)

    opentime, lopen, lhigh, llow, lclose, lvol, closetime = [], [], [], [], [], [], []
    for i in candle : 
        opentime.append(i[0])
        lopen.append(i[1])
        lhigh.append(i[2])
        llow.append(i[3])
        lclose.append(i[4])
        lvol.append(i[5])
        closetime.append(i[6])
    dfList['Open_time'] = opentime
    dfList['Close_time'] = closetime
    normal_time=[]
    normal_close_time=[]
    for i in range(len(opentime)):
        #normal_time.append(opentime[i]/1000)
        normal_time.append(datetime.fromtimestamp(opentime[i]/1000, tz=timezone.utc))
        normal_close_time.append(datetime.fromtimestamp(closetime[i]/1000, tz=timezone.utc))
        
    dfList['Open time international']=normal_time
    dfList['Close time international']=normal_close_time
    dfList['Open'] = np.array(lopen).astype(float)
    dfList['High'] = np.array(lhigh).astype(float)
    dfList['Low'] = np.array(llow).astype(float)
    dfList['Close'] = np.array(lclose).astype(float)
    dfList['Volume'] = np.array(lvol).astype(float)
    
    dfList['Mean']=(np.array(lclose).astype(float) + np.array(lopen).astype(float))/2

    df= dfList   

    return(df)




def graph_view(request):
    # Votre code de génération du graphique avec Matplotlib
    

    data=generation_data('BTCUSDT', '1h')
    ax = plt.axes()
    axis=plt.axis()
    ax.set_facecolor((0.161, 0.152, 0.157))

    #axis.set_facecolor((0.100, 0.93, 0.96))
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.plot(data['Open time international'], data['Open'], color='red', linewidth=2, mouseover=True, fillstyle='full')
    
    #liste = get_tri_par( 'market_cap', 'asc')

    #dataframe = pd.DataFrame(liste)

    

    
    # Conversion du graphique en image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encodage de l'image en base64
    graphic = urllib.parse.quote(base64.b64encode(image_png))
    return render(request, 'DocBlog/graph.html', {'graphic': graphic})






def calculator_view(request):
    if request.method == 'POST':
        expression = request.POST.get('expression', '')
        try:
            result = eval(expression)
        except Exception as e:
            result = 'Erreur: {}'.format(str(e))
    else:
        result = ''
    return render(request, 'DocBlog/calculator.html', {'result': result})








import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer # It has the ability to lemmatize.
import tensorflow as tensorF
from tensorflow import keras # A multidimensional array of elements is represented by this symbol.
from keras import Sequential, optimizers # Sequential groups a linear stack of layers into a tf.keras.Model
from keras.layers import Dense, Dropout

nltk.download("punkt")# required package for tokenization
nltk.download("wordnet")# word database


data = {"intents": [

             {"tag": "age",
              "patterns": ["how old are you?", "What's your age"],
              "responses": ["I am 2 years old and my birthday was yesterday", "I'm 18 years old wbu "]
             },
              {"tag": "greeting",
              "patterns": [ "Hi", "Hello", "Hey", "What's up", "wassup", "su^p"],
              "responses": ["Hi there", "Hello", "Hi :)", "Hey there", "Hiii"],
             },
              {"tag": "goodbye",
              "patterns": [ "bye", "later", "see you", "i'm out"],
              "responses": ["Bye", "take care", "ok see you"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["I'm Lucie what is yours" "Lucie and you ?", "Lucie and you ?"]
             },
             {"tag": "etat",
              "patterns": ["How are you?", "How are u?", "How u doin", "What's up"],
              "responses": ["Good wbu ?", "Doing good how are you ?", "Doing good what about you"]
             },
             {"tag": "location",
              "patterns": ["where do you live?", "where are you from ?"],
              "responses": ["I live in Zurich right now and u ?" "I live in zurich at the moment ", "From Zurich wbu ?"]
             }

]}


lm = WordNetLemmatizer()

def create_word(data):
  #for getting words
    # lists
    ourClasses = []
    newWords = []
    documentX = []
    documentY = []
    # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            ournewTkns = nltk.word_tokenize(pattern)# tokenize the patterns
            newWords.extend(ournewTkns)# extends the tokens
            documentX.append(pattern)
            documentY.append(intent["tag"])


        if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
            ourClasses.append(intent["tag"])

    newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
    newWords = sorted(set(newWords))# sorting words
    ourClasses = sorted(set(ourClasses))# sorting classes

    return(ourClasses, newWords, documentX, documentY )



def create_model(ourClasses, documentX,documentY, newWords):
    trainingData = [] # training list array
    outEmpty = [0] * len(ourClasses)
    # bow model
    for idx, doc in enumerate(documentX):
        bagOfwords = []
        text = lm.lemmatize(doc.lower())
        for word in newWords:
            bagOfwords.append(1) if word in text else bagOfwords.append(0)

        outputRow = list(outEmpty)
        outputRow[ourClasses.index(documentY[idx])] = 1
        trainingData.append([bagOfwords, outputRow])

    random.shuffle(trainingData)
    trainingData = num.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

    x = num.array(list(trainingData[:, 0]))# first trainig phase
    y = num.array(list(trainingData[:, 1]))
    iShape = (len(x[0]),)
    oShape = len(y[0])
    # parameter definition
    ourNewModel = Sequential()
    # In the case of a simple stack of layers, a Sequential model is appropriate

    # Dense function adds an output layer
    ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
    # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
    ourNewModel.add(Dropout(0.5))
    # Dropout is used to enhance visual perception of input neurons
    ourNewModel.add(Dense(64, activation="relu"))
    ourNewModel.add(Dropout(0.3))
    ourNewModel.add(Dense(oShape, activation = "softmax"))
    # below is a callable that returns the value to be used with no arguments
    md = optimizers.Adam(learning_rate=0.01, decay=1e-6)
    # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
    ourNewModel.compile(loss='categorical_crossentropy',
                optimizer=md,
                metrics=["accuracy"])

    ourNewModel.fit(x, y, epochs=200, verbose=0)

    return ourNewModel






def ourText(text):
  newtkns = nltk.word_tokenize(text)
  newtkns = [lm.lemmatize(word) for word in newtkns]
  return newtkns

def wordBag(text, vocab):
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns:
    for idx, word in enumerate(vocab):
      if word == w:
        bagOwords[idx] = 1
  return num.array(bagOwords)

def Pclass(text, vocab, labels, ourNewModel):
  
  bagOwords = wordBag(text, vocab)
  ourResult = ourNewModel.predict(num.array([bagOwords]))[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse=True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson):
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents:
    if i["tag"] == tag:
      ourResult = random.choice(i["responses"])
      break
  return ourResult




ourClasses, newWords, X, Y = create_word(data)
ourNewModel = create_model(ourClasses, X,Y, newWords)


messages=[]
def get_message(request) :


    

    if request.method == 'POST':
        expression = request.POST.get('expression', '')

        try:
            
            intents = Pclass(expression, newWords, ourClasses, ourNewModel)
            result = getRes(intents, data)
            #result = eval(expression)


            
            messages.append({'user': expression, 'bot': result})
            
            if len(messages)>1:
                messages.remove[0]
            # Stockez l'historique des messages dans localStorage
           # request.session['chat_messages'] = messages
        except Exception as e:
            result = 'Erreur: {}'.format(str(e))
    else:
        result = ''
    return render(request, 'DocBlog/calculator.html', {'result': result, 'messages': messages})
