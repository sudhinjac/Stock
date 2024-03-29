FROM python:3.7
COPY . ./Stock
WORKDIR /Stock
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT
CMD streamlit run port1.py --server.port $PORT
