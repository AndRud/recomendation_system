from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

title = pd.read_csv('small_dataset/movies.csv')
title.columns = ['iid', 'title', 'genres']
dataset = pd.read_csv('small_dataset/testset.csv')
users = pd.read_csv('small_dataset/users.csv', index_col = 0)
iids = dataset['uid'].unique()

@app.route('/')
def home():
	return render_template('home.html', users = users)

def predict(uid, n, dataset, predictor, title):
    iids_uid = dataset[dataset['uid'] == uid]['iid'].unique()
    iids_to_pred = np.setdiff1d(iids, iids_uid)
    test_set = [[uid, iid, 4.0] for iid in iids_to_pred]
    predict = predictor.test(test_set)
    df = pd.DataFrame(predict).drop('details', axis = 1).sort_values('est', ascending = False).head(n)
    return pd.merge(df, title, on = 'iid')

@app.route('/predict',methods=['POST'])
def pred():
    filename = 'final_model.sav'
    testset = pd.read_csv('small_dataset/testset.csv')
    predictor = pickle.load(open(filename, 'rb'))
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        films_count = int(request.form['films_count'])
        films_prediction = predict(uid = user_id, n = films_count, dataset = dataset, predictor = predictor, title = title)
        user = users['First Name'][user_id] + ' ' + users['Last Name'][user_id]
    return render_template('result.html', films_prediction = films_prediction['title'], user = user)

if __name__ == '__main__':
	app.run(debug=True)

def user_watch(uid, dataset):
    return pd.merge(dataset[dataset['uid'] == uid], title, on = 'iid').sort_values('rating', ascending = False).head(20)
