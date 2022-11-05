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
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

