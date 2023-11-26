# Use the Python 3.11 base image
FROM python:3.11

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

# Set the working directory inside the container
ENV APP_HOME /app
WORKDIR $APP_HOME

ADD requirements.txt main.py /app/

RUN pip install -r /app/requirements.txt

CMD ["python", "/app/main.py"]