from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

import json
import requests

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise import KNNBasic
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

app = Flask(__name__)
CORS(app)
print("initiated the app! With a name of:", __name__)

@app.route('/api',methods=['GET'])
def isActive():
    return jsonify(status="service is online", api_version="0.1")

algo = KNNBasic()

@app.route('/train',methods=['GET'])
def train_model():
    histories = requests.get('https://whispering-refuge-67560.herokuapp.com/api/histories')
    history_data = json.loads(histories.content.decode('utf-8'))
    data_train = pd.DataFrame.from_dict(history_data, orient='columns')
    data_train.drop(columns=['booking_id', 'createdAt', 'history_id', 'updatedAt'])
    data_train = data_train[['tid','gid','rating']]

    sim_options = {
    'name': 'cosine',
    'user_based': False
    }
 
    global algo
    algo = KNNBasic(sim_options=sim_options)
    
    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(0, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(data_train[['tid', 'gid', 'rating']], reader)

    # sample random trainset and testset
    # test set is made of 25% of the ratings.
    trainingSet = data.build_full_trainset()
    #trainset, testset = train_test_split(data, test_size=.25)

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainingSet)
    return jsonify(status="training in progress")


@app.route('/search',methods=['POST'])
def search_guide():
    data = request.get_json(force=True)
    search_city = data['city']
    user_id = data['user']



    #top3_recommendations = get_top3_recommendations(predictions)

    # i=0;
    # for User, Rating in top3_recommendations.items():
    #     print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])
    #     i=i+1;
    #     if(i==5):
    #         break;


    #accuracy.rmse(predictions)
    global algo
    pred_test = algo.predict(user_id, 0,verbose=True)
    #pred = algo.predict(34, 32)
    #return "RMSE with KNNBasic: "+ str(pred_test[0])+'-'+str(pred_test[1])+'------'+str(pred_test[3])
    return jsonify(search_city=search_city, predicted_rating=pred_test[3])

@app.route('/test',methods=['GET'])
def api_test():
    #req = requests.get('https://whispering-refuge-67560.herokuapp.com/api/guides?filter={"where": {"keywords": {"fav_places": ["san_francisco"]}}}')
    #return jsonify(req.content.decode('utf-8'))
    return jsonify(get_guides("san_francisco"))

def get_guides(city):
    guide_list = []
    response = requests.get('https://whispering-refuge-67560.herokuapp.com/api/guides?filter={"where": {"keywords": {"fav_places": ["'+city+'"]}}}')
    data = json.loads(response.content.decode('utf-8'))
    for guide in data:
        guide_list.append(guide["gid"])
    #json.dumps([dict(gid=gd) for gd in data])

    if response.status_code == 200:
        return guide_list
    else:
        return None

def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

if __name__ == '__main__':
    app.run(port=5000, debug=True)
