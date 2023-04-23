FROM python:3.8
RUN pip install boto3 torch torchvision
COPY . /app
WORKDIR /app
CMD ["python", "train_cifar10.py"]
