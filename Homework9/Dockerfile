FROM python:3.10-slim-bullseye

# copy lambda function
COPY lambda_function.py .

# install dependencies
RUN pip install --upgrade pip && \
    pip install numpy==1.23.5 pillow tensorflow==2.14.0 && \
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

CMD ["lambda_function.lambda_handler"]
