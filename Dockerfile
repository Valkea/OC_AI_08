FROM python:3.8.12-slim

ENV PORT=5000
EXPOSE 5000

# start to install backend-end stuff
RUN mkdir -p /app
WORKDIR /app

# Install Python requirements.
COPY requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt

# Install Python requirements.
COPY ["API_server.py", "./"]
COPY ["models/FPN-efficientnetb7_with_data_augmentation_2_diceLoss_512x256.tflite", "./models/"]
COPY ["data/preprocessed/512x256/val", "./data/preprocessed/512x256/val"]

# Start server
#ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:$PORT", "API_server:app"]
CMD gunicorn API_server:app --bind 0.0.0.0:$PORT --timeout=60 --threads=2
