# from flask import Flask,render_template
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#    return render_template("main.html")

# @app.route('/predata')
# def preddata(test_data):
#    scaler = StandardScaler()
#    test_data_scaled = (test_data.iloc[:,:-1].values - scaler.mean_)/np.sqrt(scaler.var_)
#    pd.DataFrame(test_data_scaled).describe()
#    loaded_model = pickle.load(open("E:\project\sample project\svmoncrwu_model", "rb"))
#    result=loaded_model.predict(test_data_scaled)
#    return render_template("preddata.html")


# if __name__ == '__main__':
#   app.run(debug=True)
from flask import Flask, render_template, request
import csv
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload():
    
    data=[]
    file = request.files['file']
    file_path = 'uploads/' + file.filename  # Construct the file path
    file.save(file_path)  # Save the file to disk

    with open(file_path, 'rt') as f:  # Open the file for reading
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
            # Process the row data
    # Render the template with the data
    return render_template('data.html', data=data)




@app.route('/test',methods=['POST'])
def test():
    UPLOAD_FOLDER = os.path.abspath(r"E:\project\sample project\uploads")
    csv_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        scaler = StandardScaler()
        test_data_scaled=scaler.fit_transform(df.iloc[:,:-1])
        
        pd.DataFrame(test_data_scaled).describe()
        predictions = run_ml_model(test_data_scaled)
    return render_template('predictions.html', predictions=predictions)

    
    
        
    


def run_ml_model(test_data_scaled):
    loaded_model = pickle.load(open("E:\project\sample project\svmoncrwu_model", "rb"))
    predictions=loaded_model.predict(test_data_scaled)
    return predictions

@app.route('/Home',methods=['POST'])
def Home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
