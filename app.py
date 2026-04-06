from flask import Flask, request, Response
import pandas as pd
import cross_attn_with_mlp_predict
import lightGBT_app_predict
import os
import logging
import argparse

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


@app.route('/nfl/health')
def health_check():
    return 'dont worry i m there'

@app.route('/nfl/cross_attn_with_mlp_predict_play', methods=['POST']) 
def cross_attn_with_mlp_predict_play():
    csv_file = request.files['file'] 
    df = pd.read_csv(csv_file.stream)
    res_df = cross_attn_with_mlp_predict.predict(df)

    return  Response(
        res_df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cross_attn_with_mlp_predict.csv"}
    )

@app.route('/nfl/lightGBT_predict_play', methods=['POST']) 
def lightGBT_predict_play():
    csv_file = request.files['file'] 
    df = pd.read_csv(csv_file.stream) 

    res_df = lightGBT_app_predict.predict(df)

    return  Response(
        res_df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=lightGBT_predict.csv"}
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the FastAPI application.")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the app on")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)