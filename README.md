# Build image
```
docker build -t mentee/aws-s3-docker-example .
```

# Run example
Replace key and secret in the following command:
```
docker run --rm --gpus all  -e AWS_ACCESS_KEY_ID=[AWS ACCESS KEY ID] -e AWS_SECRET_ACCESS_KEY=[AWS SECRET ACCESS KEY] -it mentee/aws-s3-docker-example
```

Use `--gpus '"device=4"'` to select a specific GPU.
