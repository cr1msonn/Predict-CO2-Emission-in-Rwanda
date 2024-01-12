from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import HTTPException
import pandas as pd
from xgboost import XGBRegressor
from pydantic import BaseModel, constr
import json
from datetime import datetime
import logging
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from preprocces import DataProcessor

logger = logging.getLogger("fastaoi_logger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)    

logger.addHandler(file_handler)

train = pd.read_csv('train.csv')
train = train[['latitude', 'longitude', 'year', 'week_no', 'Ozone_solar_azimuth_angle', 'emission']]

processor = DataProcessor(train)



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def load_model():
    loaded_model = XGBRegressor()
    loaded_model.load_model('XGB_4')

    return loaded_model


class PredictionResult(BaseModel):
    prediction: float

class InputData(BaseModel):
    date: str
    latitude: float
    longitude: float
    Ozone_solar_azimuth_angle: float

def convert_to_datetime(self):
    date_obj = datetime.strptime(self.date, "%Y-%m-%d")
    date = date_obj.strftime("%Y-%m-%d")
    return date

# Allow OPTIONS method for the /predict endpoint
@app.options("/predict")
def options_predict():
    return {"msg": "OK"}


@app.post("/predict")
async def predict_emission(data: InputData):
    try:
    # Convert the datetime string to a datetime object

        date = data.date
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        week_number =date_obj.strftime("%U")
        year = date_obj.strftime("%Y")

        data_dict = {'date': [date], 'latitude': [data.latitude], 'longitude': [data.longitude], 'year': [year], 'week_no': [week_number],
        'Ozone_solar_azimuth_angle': [data.Ozone_solar_azimuth_angle]}
        data_df = pd.DataFrame(data_dict)

        data_df['year'] = data_df['year'].astype(int)
        data_df['week_no'] = data_df['week_no'].astype(int)
        model = load_model()    
       
        processed_data = processor.preprocess(data_df)

        model_input = processed_data[[
            'latitude',
            'longitude',
            'year',
            'week_no',
            'Ozone_solar_azimuth_angle',
            'week_no_sin',
            'week_no_cos',
            'month',
            'rot_45_x',
            'rot_45_y',
            'rot_30_x',
            'rot_30_y',
            'distance_to_max_emission'
        ]]

        preds = model.predict(model_input)
        predictions_result = {"predictions": float(preds[0])}
        return predictions_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")

# async def main():

#     content =  """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <meta http-equiv="X-UA-Compatible" content="IE=edge" />

#     <script src="https://polyfill.io/v3/polyfill.min.js?features=default"></script>
#     <title>Emission Prediction Form</title>
#     <style>
#         body {
#             font-family: 'Arial', sans-serif;
#             background-color: #f4f4f4;
#             margin: 0;
#             padding: 0;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             height: 100vh;
#         }

#         .form-container {
#             background-color: #fff;
#             border-radius: 8px;
#             box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
#             padding: 20px;
#             width: 400px;
#         }

#         .form-column {
#             display: flex;
#             flex-direction: column;
#             margin-bottom: 15px;
#         }

#         .form-column label {
#             margin-bottom: 5px;
#             font-weight: bold;
#         }

#         .form-column input {
#             padding: 8px;
#             border: 1px solid #ccc;
#             border-radius: 4px;
#         }

#         .predict-button {
#             background-color: #007bff;
#             color: #fff;
#             padding: 10px;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#         }

#         .prediction-result {
#             margin-top: 15px;
#             font-weight: bold;
#             color: #28a745;
#         }
#     </style>
# </head>
# <body>
#     <div id="map"></div>

#     <!-- 
#       The `defer` attribute causes the callback to execute after the full HTML
#       document has been parsed. For non-blocking uses, avoiding race conditions,
#       and consistent behavior across browsers, consider loading using Promises.
#       See https://developers.google.com/maps/documentation/javascript/load-maps-js-api
#       for more information.
#       -->
#     <script
#       src="https://maps.googleapis.com/maps/api/js?key=AIzaSyB41DRUbKWJHPxaFjMAwdrzWzbVKartNGg&callback=initMap&v=weekly"
#       defer
#     ></script>

#     <div class="form-container">
#         <div class="form-column">
#             <label for="date">Date:</label>
#             <input type="text" id="date" placeholder="Enter date" />
#         </div>
#         <div class="form-column">
#             <label for="latitude">Latitude:</label>
#             <input type="number" id="latitude" placeholder="Enter latitude" />
#         </div>
#         <div class="form-column">
#             <label for="longitude">Longitude:</label>
#             <input type="number" id="longitude" placeholder="Enter longitude" />
#         </div>
#         <div class="form-column">
#         <label for="ozone">Ozone Solar Azimuth Angle:</label>
#         <input type="range" id="ozone" min="-360" max="360" step="1" value="180" oninput="updateOzoneValue()">
#         <span id="ozoneValue">180</span>
#         </div>

#         <button class="predict-button" onclick="predictEmission()">Predict</button>
#         <div class="prediction-result" id="predictionResult"></div>
#     </div>
    


#     <script>
#     async function predictEmission() {
#         // Get input values from the form
#         const date = document.getElementById("date").value;
#         const latitude = parseFloat(document.getElementById("latitude").value);
#         const longitude = parseFloat(document.getElementById("longitude").value);
#         const ozone = parseFloat(document.getElementById("ozone").value);

#         // Make a POST request to your FastAPI endpoint
#         const response = await fetch('http://127.0.0.1:8000/predict', {
#             method: 'POST',
#             headers: {
#                 'Content-Type': 'application/json',
#             },
#             body: JSON.stringify({
#                 date: date,
#                 latitude: latitude,
#                 longitude: longitude,
#                 Ozone_solar_azimuth_angle: ozone
#             }),
#         });

#         // Parse the response
#         const result = await response.json();

#         // Display the prediction result on the screen
#         const predictionResultElement = document.getElementById("predictionResult");
#         predictionResultElement.innerHTML = `Predicted Emission: ${result.predictions}`;
#     }
#     </script>
#     <script>
#     function updateOzoneValue() {
#         const ozoneSlider = document.getElementById("ozone");
#         const ozoneValue = document.getElementById("ozoneValue");
#         ozoneValue.innerText = ozoneSlider.value;
#     }

#     // Call this function initially to set the default value
#     updateOzoneValue();
#     </script>
#     """

#     return HTMLResponse(content=content)
