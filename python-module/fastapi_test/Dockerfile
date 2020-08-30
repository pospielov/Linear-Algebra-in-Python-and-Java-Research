FROM python:3

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host=0.0.0.0 --workers=8 --port=3000