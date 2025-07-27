"""
S3 model loading functionality for production deployment
"""

from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import settings


class S3ModelLoader:
    """Base class for loading models from S3"""

    def __init__(self):
        self.s3_client = None
        self._initialize_s3_client()

    def _initialize_s3_client(self):
        """Initialize S3 client with credentials"""
        try:
            # Try to use AWS credentials from environment or IAM role
            if settings.s3_access_key_id and settings.s3_secret_access_key:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=settings.s3_access_key_id,
                    aws_secret_access_key=settings.s3_secret_access_key,
                    region_name=settings.s3_region,
                )
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.s3_client = boto3.client("s3", region_name=settings.s3_region)
        except NoCredentialsError:
            self.s3_client = None

    def download_model_from_s3(
        self, model_type: str, local_dir: str, s3_key_prefix: Optional[str] = None
    ) -> bool:
        """
        Download model files from S3 to local directory

        Args:
            model_type: Type of model ("distilbert" or "distilbert-onnx")
            local_dir: Local directory to save model files
            s3_key_prefix: Optional S3 key prefix (defaults to model_type)

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            return False

        if not s3_key_prefix:
            s3_key_prefix = model_type

        try:
            # Create local directory if it doesn't exist
            Path(local_dir).mkdir(parents=True, exist_ok=True)

            # List objects in S3 bucket with the prefix
            response = self.s3_client.list_objects_v2(
                Bucket=settings.s3_bucket_name,
                Prefix=f"{settings.s3_model_prefix}/{s3_key_prefix}/",
            )

            if "Contents" not in response:
                return False

            # Download each file
            for obj in response["Contents"]:
                s3_key = obj["Key"]
                local_file = Path(local_dir) / Path(s3_key).name

                self.s3_client.download_file(
                    settings.s3_bucket_name, s3_key, str(local_file)
                )

            return True

        except ClientError as e:
            print(f"Error downloading model from S3: {e}")
            return False

    def upload_model_to_s3(
        self, local_dir: str, model_type: str, s3_key_prefix: Optional[str] = None
    ) -> bool:
        """
        Upload model files from local directory to S3

        Args:
            local_dir: Local directory containing model files
            model_type: Type of model for S3 key prefix
            s3_key_prefix: Optional S3 key prefix (defaults to model_type)

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            return False

        if not s3_key_prefix:
            s3_key_prefix = model_type

        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                return False

            # Upload each file in the directory
            for file_path in local_path.iterdir():
                if file_path.is_file():
                    s3_key = (
                        f"{settings.s3_model_prefix}/{s3_key_prefix}/{file_path.name}"
                    )

                    self.s3_client.upload_file(
                        str(file_path), settings.s3_bucket_name, s3_key
                    )

            return True

        except ClientError as e:
            print(f"Error uploading model to S3: {e}")
            return False

    def model_exists_in_s3(
        self, model_type: str, s3_key_prefix: Optional[str] = None
    ) -> bool:
        """
        Check if model files exist in S3

        Args:
            model_type: Type of model
            s3_key_prefix: Optional S3 key prefix (defaults to model_type)

        Returns:
            True if model exists in S3, False otherwise
        """
        if not self.s3_client:
            return False

        if not s3_key_prefix:
            s3_key_prefix = model_type

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=settings.s3_bucket_name,
                Prefix=f"{settings.s3_model_prefix}/{s3_key_prefix}/",
                MaxKeys=1,
            )

            return "Contents" in response and len(response["Contents"]) > 0

        except ClientError:
            return False
