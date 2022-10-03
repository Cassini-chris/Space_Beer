from flask import Flask
from flask import jsonify
import numpy as np
from flask import request, render_template
import io
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from datetime import datetime
import os
from google.cloud import storage
import tempfile
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

app = Flask(__name__)

@app.route('/')
def index():
    #return("test")
    return render_template('web-interface.html')

storage_client = storage.Client()

print(" * Loading App ")

@app.route("/predicto", methods = ["POST"])

def predicto():
    message = request.get_json(force=True)
    encoded = message['image']
    api_endpoint = "us-central1-aiplatform.googleapis.com"
    project="163870460651"
    endpoint_id="3586558551293689856"
    location="us-central1"
    filename=encoded

    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    encoded_content = filename

    instance = predict.instance.ImageClassificationPredictionInstance(content=encoded_content,).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(confidence_threshold=0.5, max_predictions=5, ).to_value()
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    predictions = response.predictions
    print(predictions)
    for prediction in predictions:
        final_output = dict(prediction)
        print(final_output)
    print(type(final_output))

    for key, value in final_output.items() :
      print(key, value)
    confi = (final_output['confidences'])
    print(confi[0])

    final_response =confi[0]
    prediction_result = final_response

    response = {'prediction': {
            'level_1':prediction_result,
            'level_2':prediction_result,
            'level_3':prediction_result,
            'level_4':prediction_result,
            'level_5':prediction_result,
            'level_6':prediction_result,
            'level_7':prediction_result,
            'level_8':prediction_result,
            'level_9':prediction_result,
            'level_10':prediction_result

    }}
    print(response)
    return jsonify(response)
