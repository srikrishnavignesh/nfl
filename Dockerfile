FROM python:3.14.3-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt


COPY gru_scaler.joblib app.py  gru_app_predict.py lightGBT_app_predict.py gru.py lightGBT.py ./
COPY cross_attn_best_weight_dx.pt cross_attn_best_weight_dy.pt ./
COPY lightGBT_dx_model.txt lightGBT_dy_model.txt ./

CMD ["python3", "app.py", "--port", "7860"]
