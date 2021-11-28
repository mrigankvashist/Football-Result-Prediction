# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pickle import load

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))

Standings=pd.read_csv('/Users/KillSwitch/Desktop/footy/eplstandings.csv')
Standings=Standings.set_index('Team')
encodings=pd.read_csv('/Users/KillSwitch/Desktop/footy/encodings.csv')
encodings=encodings.set_index('Team').T.to_dict('list')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    home_array = list()
    away_array=list()
    
    if request.method == 'POST':
        
        home_team = request.form['home-team']
        home_array=home_array+ encodings[home_team]
            
    
        away_team = request.form['away-team']
        away_array=away_array + encodings[away_team]
        
        hthg = int(request.form['hthg'])
        athg = int(request.form['athg'])
        htp = int(request.form['htp'])
        atp = int(request.form['atp'])
        diff_pts=htp-atp
        ht_lp=Standings.loc[home_team,'2021']
        at_lp=Standings.loc[away_team,'2021']
        diff_lp=ht_lp-at_lp
        
        temp_array =  [hthg, athg, htp, atp,ht_lp,at_lp,diff_pts,diff_lp] +home_array +away_array
        scaled=scaler.transform(np.array([[hthg, athg, htp, atp,ht_lp,at_lp,diff_pts,diff_lp]]))
        data=np.concatenate((scaled.flatten(),np.array(home_array),np.array(away_array)))
        data=data.reshape(1,-1)
        pred=pd.DataFrame(model.predict_proba(data))
        away = round(pred.iloc[0,0],3)
        draw=round(pred.iloc[0,1],3)
        home=round(pred.iloc[0,2],3)
              
        return render_template('result.html', home=home,away=away,draw=draw)



if __name__ == '__main__':
	app.run(debug=True)
