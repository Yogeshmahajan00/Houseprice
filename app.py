from flask import Flask,request,app,url_for,render_template
import numpy as np
import numpy as pd
import pickle

app=Flask(__name__)

regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    output=regmodel.predict(final_input)[0] ######[0]--to get only values inside the array
    return render_template('home.html',prediction_text="The house price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)