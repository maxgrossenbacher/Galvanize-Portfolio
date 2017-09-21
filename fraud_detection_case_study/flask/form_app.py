from flask import Flask, render_template, request, jsonify
import pickle
import json
import requests
import socket
import time
from datetime import datetime
import re
import pandas as pd
import ast
import os
from flask_sqlalchemy import SQLAlchemy
import re
from src.connect_database import connect_psql
import src.data_cleaning_pipeline as dcp


app = Flask(__name__)
PORT = 5353
REGISTER_URL = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
DATA = []
TIMESTAMP = []
#
# with open('../src/model.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('form/index.html')

@app.route('/score', methods=['GET','POST'])
def score():
    m_zero=0
    request.method == "POST"
    # for _ in range(1):
        # get url that the user has entered
    url = REGISTER_URL
    r = requests.get(url)
    m = re.search('(?<="object_id": )\w+', r.text)
    if m.group(0) != m_zero:
        DATA.append(json.loads(re.sub(r"\s+", " ", r.text)))
        print(type(json.loads(re.sub(r"\s+", " ", r.text))))
        # DATA.append(json.dumps(re.sub(r"\s+", " ", r.text), sort_keys=True, indent=4, separators=(',', ': ')))
        # print(type(json.dumps(re.sub(r"\s+", " ", r.text), sort_keys=True, indent=4, separators=(',', ': ')))
        TIMESTAMP.append(time.time())
        m_zero = m.group(0)
    print(len(DATA))
    time.sleep(2)
    for row in DATA:
        pipeline = dcp.Data_Cleaning(row)
        risk, probability = pipeline.run_pipeline()
            # df = pipeline.df
            # df['fraud_predictions'] = probability
    #
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    r = requests.get(REGISTER_URL)
    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>')
        almost = re.sub(cleanr, '', raw_html)
        cleantext = re.sub(r'\s+', ' ', almost, flags=re.UNICODE)
        return cleantext
    new = cleanhtml(r.text)
    # df = pd.read_json(new)
    # r = json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ': '))
    # new = model(r)
    event_dict = ast.literal_eval(new)
    org = "Is '"+str(event_dict['org_name'])+"' up to no good?"
    return render_template('form/score.html', org_name = org, risk=risk, probability=probability, object_id=m.group(0))#, article=data, predicted=pred)


def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


if __name__ == '__main__':
    ip_address = socket.gethostbyname("")
    print("attempting to register %s:%d" % (ip_address, PORT))
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
    #
    # my_ip = socket.gethostbyname("")
    # my_port = 5000
    # register_for_ping(my_ip, my_port)
    # app.run(host='0.0.0.0', debug=False, threaded=True)


# from flask import Flask, render_template, request, jsonify
# import pickle
# import json
# import requests
# import socket
# import time
# from datetime import datetime
# import re
# import pandas as pd
# import ast
#
#
# app = Flask(__name__)
# PORT = 5000
# # REGISTER_URL = "http://10.3.0.79:5000/register"
# REGISTER_URL = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
# DATA = []
# TIMESTAMP = []
#
# with open('../src/model.pkl', 'rb') as f:
#     model = pickle.load(f)
#
# @app.route('/', methods=['GET'])
# def index():
#     """Render a simple splash page."""
#     return render_template('form/index.html')
#
# @app.route('/score', methods=['POST'])
# def score():
#     """Recieve live stream data to be classified from an input form and use the
#     model to classify.
#     """
#     url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
#     r = requests.get(REGISTER_URL)
#     def cleanhtml(raw_html):
#         cleanr = re.compile('<.*?>')
#         almost = re.sub(cleanr, '', raw_html)
#         cleantext = re.sub(r'\s+', ' ', almost, flags=re.UNICODE)
#         return cleantext
#     new = cleanhtml(r.text)
#     # df = pd.read_json(new)
#     # r = json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ': '))
#     # new = model(r)
#     event_dict = ast.literal_eval(new)
#     org = "Is '"+str(event_dict['org_name'])+"' up to no good?"
#     prediction = 'Probably'
#
#     return render_template('form/score.html', org_name=org, pred=prediction)#, article=data, predicted=pred)
#
#
# def register_for_ping(ip, port):
#     registration_data = {'ip': ip, 'port': port}
#     requests.post(REGISTER_URL, data=registration_data)
#
#
# if __name__ == '__main__':
#     my_ip = socket.gethostbyname("")
#     my_port = 5000
#     register_for_ping(my_ip, my_port)
#     app.run(host='0.0.0.0', debug=False, threaded=True)
