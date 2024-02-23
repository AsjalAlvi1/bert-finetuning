FROM tiangolo/uwsgi-nginx-flask:python3.9
COPY . /app
WORKDIR /app
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1EpqO4SkZ9EWsIw4qDl1UGQ4dJq-LdpSD
RUN ls -lah
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 8080
CMD ["uvicorn", "Bert_API.py:app", "--host", "0.0.0.0", "--port", "8080"]
