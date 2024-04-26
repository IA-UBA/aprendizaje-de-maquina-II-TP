import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary

def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "water_quality_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass

class ModelInput(BaseModel):
    """
    Input schema for the water quality prediction model.

    This class defines the input fields required by the water quality prediction model along with their descriptions
    and validation constraints.

    :param ph: Indicator of acidic or alkaline condition of water status.
    :param Hardness:Originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.
    :param Solids: (Total dissolved solids - TDS) Indicates the amount of minerals in water (mg/l).
    :param Chloramines: Amout of major disinfectant used in public water systems (mg/L or ppm).
    :param Sulfate: Amout of naturally occurring substance that is found in minerals, soil, and rocks (mg/L).
    :param Conductivity: Electrical conductivity. Measures the ionic process of a solution that enables it to transmit current (μS/cm).
    :param Organic_carbon: Total amount of carbon in organic compounds in pure water (mg/L).
    :param Trihalomethanes: Amount of chemicals which may be found in water treated with chlorine (ppm).
    :param Turbidity: Quantity of solid matter present in the suspended state (NTU).
    """

    ph: float = Field(
        description="Indicator of acidic or alkaline condition of water status.",
        ge=0,
    )
    Hardness: float = Field(
        description="Originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.",
        ge=0,
    )
    Solids: float = Field(
        description="(Total dissolved solids - TDS) Indicates the amount of minerals in water (mg/l).",
        ge=0,
    )
    Chloramines: float = Field(
        description="Amout of major disinfectant used in public water systems (mg/L or ppm).",
        ge=0,
    )
    Sulfate: float = Field(
        description="Amout of naturally occurring substance that is found in minerals, soil, and rocks (mg/L).",
        ge=0,
    )
    Conductivity: float = Field(
        description="Electrical conductivity. Measures the ionic process of a solution that enables it to transmit current (μS/cm).",
        ge=0,
    )
    Organic_carbon: float = Field(
        description="Total amount of carbon in organic compounds in pure water (mg/L).",
        ge=0,
    )
    Trihalomethanes: float = Field(
        description="Amount of chemicals which may be found in water treated with chlorine (ppm).",
        ge=0,
    )
    Turbidity: float = Field(
        description="Quantity of solid matter present in the suspended state (NTU).",
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ph": 7.1365540741406654,
                    "Hardness": 204.8904554713363,
                    "Solids": 20791.318980747023,
                    "Chloramines":7.300211873184757,
                    "Sulfate": 368.51644134980336,
                    "Conductivity": 564.3086541722439,
                    "Organic_carbon": 10.3797830780847,
                    "Trihalomethanes": 86.9909704615088,
                    "Turbidity": 2.9631353806316407,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the water quality prediction model.

    This class defines the output fields returned by the water quality prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if the water is potable.
    :param str_output: Output of the model in string form. Can be "Potable water" or "Non-potable water".
    """

    int_output: bool = Field(
        description="Output of the model. True if the water is potable",
    )
    str_output: Literal["Potable water", "Non-potable water"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Potable water",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("water_quality_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Water Quality Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Water Quality Detector API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting water quality.

    This endpoint receives features related to a water sample and predicts whether the water is potable
    or not using a trained model. It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Scale the data using standard scaler
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Non-potable water"
    if prediction[0] > 0:
        str_pred = "Potable water"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)