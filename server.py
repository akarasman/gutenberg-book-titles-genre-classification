import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request
import os, time
from keras.preprocessing import sequence


app = Flask(__name__)
cnn_model = pickle.load(open('models/cnn_model.pickle', 'rb'))
ann_model = pickle.load(open('models/ann_model.pickle', 'rb'))
lstm_model = pickle.load(open('models/lstm_model.pickle', 'rb'))

le = pickle.load(open("transforms/encoder.pickle", 'rb'))
token = pickle.load(open("transforms/tokenizer.pickle", 'rb'))
vect = pickle.load(open("transforms/vectorizer.pickle", 'rb'))

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index():
    book = request.form['input_data']
    # Filter book title data 
    book = book.lower()
    book = book.split()
    ps = PorterStemmer()
    book = [ps.stem(word) for word in book if not word in set(stopwords.words('english'))]
    book = ' '.join(book)
    
    # Load previously fitted models
    model = request.form['model']
    if(model == "models/cnn_model.pickle") :
        clf = cnn_model
    elif(model == "models/ann_model.pickle") :
        print("in ann")
        clf = ann_model
    elif(model == "models/lstm_model.pickle") :
        clf = lstm_model
 
# Boolean value encodes whether or not model is made with keras api
    keras_model = model == 'models/cnn_model.pickle' or \
                  model == 'models/ann_model.pickle' or \
                  model == 'models/lstm_model.pickle'
            
# Extract class probabilities
    if(keras_model):
        tokenizer = token
        x = sequence.pad_sequences(tokenizer.texts_to_sequences([book]), maxlen=100)
        probs = clf.predict_proba(x).reshape((32,)) 
    else :
        vectorizer = vect
        probs = clf.predict_proba(vectorizer.transform([book])).reshape((32,))
     
    
    # Create class to probability mappings
    prob_classes = {k: v for k, v in zip(le.classes_, probs)}
    prob_classes = OrderedDict(reversed(sorted(prob_classes.items(), key=lambda x: x[1])[-10:]))
    
    
    # Plot class probabilities bar graph
    plt.bar(range(len(prob_classes)), list(prob_classes.values()), align='center', color=['orchid'], edgecolor='black')
    plt.xticks(range(len(prob_classes)), list(prob_classes.keys()), rotation=70)
    plt.tight_layout()
    ax = plt.axes()
    # Setting the background color
    ax.set_facecolor("black")
    
    new_graph_name = "graph" + str(time.time()) + ".png"

    for filename in os.listdir('static/'):
        if filename.startswith('graph'):  # not to remove other images
            os.remove('static/' + filename)
            
    plt.savefig('static/' + new_graph_name)
    
    # Print top 5 categories
    cat1 = list(prob_classes.keys())[0]
    prob1 = round(list(prob_classes.values())[0],2)
    cat2 = list(prob_classes.keys())[1]
    prob2 = round(list(prob_classes.values())[1],2)
    cat3 = list(prob_classes.keys())[2]
    prob3 = round(list(prob_classes.values())[2],2)
    cat4 = list(prob_classes.keys())[3]
    prob4 = round(list(prob_classes.values())[3],2)
    cat5 = list(prob_classes.keys())[4]
    prob5 = round(list(prob_classes.values())[4],2)
    return render_template('result.html', cat1 = cat1, prob1 = prob1,cat2 = cat2, prob2 = prob2,cat3 = cat3, prob3 = prob3,cat4 = cat4, prob4 = prob4,cat5 = cat5, prob5 = prob5, graph=new_graph_name)
    

if __name__ =="__main__":
    app.run(threaded = False)
        
    