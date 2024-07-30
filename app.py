from flask import Flask, jsonify, request
from torch import nn
import torch

from src.data.data_ingestion import DataIngestion
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline


app = Flask(__name__)


@app.route("/run_train", methods=["GET"])
def run_train():
    TrainPipeline().start_training()
    return (jsonify({"message": "Success Training"}), 200)


@app.route("/run_test", methods=["GET"])
def run_test():
    TrainPipeline().get_test_stats()
    return (jsonify({"message": "Success"}), 200)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        predictor = PredictPipeline(image=file)
        value = predictor.predict()
        # class_name = ("Not hotdog", "Hotdog")[value]
        return jsonify({"class": value})


if __name__ == "__main__":
    app.run(debug=True)
