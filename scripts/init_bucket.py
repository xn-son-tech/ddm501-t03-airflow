"""
Create the artifact bucket, then exit.

MinIO starts empty. MLflow will not create the bucket for you: it assumes the
bucket exists and fails on the first log_model if it does not -- after the
model has already trained.
"""
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

BUCKET = os.environ["MLFLOW_BUCKET"]
ENDPOINT = os.environ["MLFLOW_S3_ENDPOINT_URL"]


def main() -> None:
    s3 = boto3.client("s3", endpoint_url=ENDPOINT)

    # MinIO answers its port before it is ready to serve. Retry rather than
    # depend on a healthcheck inside an image we do not control.
    for attempt in range(1, 31):
        try:
            s3.list_buckets()
            break
        except (EndpointConnectionError, ClientError) as exc:
            print(f"waiting for {ENDPOINT} ({attempt}/30): {type(exc).__name__}")
            time.sleep(2)
    else:
        sys.exit(f"{ENDPOINT} never answered")

    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if BUCKET in existing:
        print(f"bucket {BUCKET} already there")
        return
    s3.create_bucket(Bucket=BUCKET)
    print(f"created bucket {BUCKET}")


if __name__ == "__main__":
    main()
