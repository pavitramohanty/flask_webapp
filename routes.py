import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np

from utils import onehotCategorical

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        entered_li = []
        # YOUR CODE FOR PART 2.2
        # get request values
        month = int(request.form['Month'])
        day = int(request.form['Day'])
        promo = int(request.form['Promo'])
        promo2 = int(request.form['Promo2'])
        stateH = int(request.form['StateH'])
        schoolH = int(request.form['SchoolH'])
        assortment = int(request.form['Assortment'])
        storeType = int(request.form['StoreType'])
        store = int(request.form['store'])

        # one-hot encode categorical variables
        stateH_encode = onehotCategorical(stateH, 4)
        assortment_encode = onehotCategorical(assortment, 3)
        storeType_encode = onehotCategorical(storeType, 4)
        store_encode = onehotCategorical(store, 1115, store=1)

        # manually specify competition distance
        comp_dist = 5458.1

        # engineer 1 observation for prediction
        # YOUR CODE START HERE
        entered_li.extend(store_encode)
        entered_li.extend(storeType_encode)
        entered_li.extend(assortment_encode)
        entered_li.extend(stateH_encode)
        entered_li.extend([comp_dist])
        entered_li.extend([promo2])
        entered_li.extend([promo])
        entered_li.extend([day])
        entered_li.extend([month])
        entered_li.extend([schoolH])

        prediction = model.predict(np.array(entered_li).reshape(1, -1))
        #prediction = model.predict(entered_li.values.reshape(1, -1))
        label = str(np.squeeze(prediction.round(2)))

        return render_template('index.html', label=label)

if __name__ == '__main__':
    # load ML model
    model = joblib.load('lr.pkl')
    # start API
    app.run()
