FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install boto3
COPY . /app
WORKDIR /app
CMD ["python", "train_cifar10.py"]
