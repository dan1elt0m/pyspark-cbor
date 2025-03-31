import base64
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from pyspark.sql.datasource import InputPartition, DataSource, DataSourceReader
from pyspark.sql.types import StructType
from typing import Iterator, Tuple, List


@dataclass
class CBORInputPartition(InputPartition):
    file_name: str


class CBORDataSource(DataSource):
    """
    An example data source for batch query using the `cbor2` library.
    """

    @classmethod
    def name(cls):
        return "cbor"

    def reader(self, schema: StructType):
        return CBORDataSourceReader(schema, self.options)


class CBORDataSourceReader(DataSourceReader):
    def __init__(self, schema, options):
        self.schema: StructType = schema
        self.options = options

    def partitions(self) -> List[CBORInputPartition]:
        """This method returns a list of InputPartition objects.

        Each InputPartition object represents a file that can be read concurrently.
        """
        path = self.options["path"]

        if path.startswith("s3://"):
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "Boto3 is required to read this file. pip install pyspark-cbor[aws]"
                )
            # Handle S3 path
            s3 = boto3.client("s3")
            parsed_url = urlparse(path)
            bucket = parsed_url.netloc
            prefix = parsed_url.path.lstrip("/")
            if prefix.endswith("/"):
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                file_names = [
                    f"s3://{bucket}/{item['Key']}"
                    for item in response.get("Contents", [])
                    if not item["Key"].endswith("/")
                ]
            else:
                file_names = [path]

        elif path.startswith("gs://"):
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required to read this file. pip install pyspark-cbor[gcp]"
                )
            # Handle GCS path
            client = storage.Client()
            parsed_url = urlparse(path)
            bucket = client.bucket(parsed_url.netloc)
            prefix = parsed_url.path.lstrip("/")
            if prefix.endswith("/"):
                blobs = bucket.list_blobs(prefix=prefix)
                file_names = [
                    f"gs://{parsed_url.netloc}/{blob.name}"
                    for blob in blobs
                    if not blob.name.endswith("/")
                ]
            else:
                file_names = [path]
        elif path.startswith("https://") and "blob.core.windows.net" in path:
            try:
                from azure.storage.blob import BlobServiceClient
            except ImportError:
                raise ImportError(
                    "azure-storage-blob is required to read this file. pip install pyspark-cbor[azure]"
                )
            # Handle Azure Blob Storage path
            parsed_url = urlparse(path)
            account_name = parsed_url.netloc.split(".")[0]
            container_name = parsed_url.path.split("/")[1]
            prefix = "/".join(parsed_url.path.split("/")[2:])
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=self.options.get("azure_credential"),
            )
            container_client = blob_service_client.get_container_client(container_name)
            if prefix.endswith("/"):
                blobs = container_client.list_blobs(name_starts_with=prefix)
                file_names = [
                    f"https://{account_name}.blob.core.windows.net/{container_name}/{blob.name}"
                    for blob in blobs
                    if not blob.name.endswith("/")
                ]
            else:
                file_names = [path]

        elif os.path.isdir(path):
            # Handle local directory
            file_names = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]

        else:
            # Single file
            file_names = [path]

        return [
            CBORInputPartition(file_name=file_name.strip()) for file_name in file_names
        ]

    def read(self, partition: CBORInputPartition) -> Iterator[Tuple]:
        """This method reads the data from the given InputPartition."""
        # Import here to serialize the modules into the workers
        import cbor2
        from pyspark_cbor.parsers import _parse_array, convert_to_struct # noqa: F401

        file_name = partition.file_name

        if file_name.startswith("s3://"):
            import boto3

            s3 = boto3.client("s3")
            parsed_url = urlparse(file_name)
            bucket = parsed_url.netloc
            key = parsed_url.path.lstrip("/")
            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response["Body"].read()
        elif file_name.startswith("gs://"):
            from google.cloud import storage

            client = storage.Client()
            parsed_url = urlparse(file_name)
            bucket = client.bucket(parsed_url.netloc)
            blob = bucket.blob(parsed_url.path.lstrip("/"))
            file_content = blob.download_as_bytes()
        elif file_name.startswith("https://") and "blob.core.windows.net" in file_name:
            from azure.storage.blob import BlobServiceClient

            parsed_url = urlparse(file_name)
            account_name = parsed_url.netloc.split(".")[0]
            container_name = parsed_url.path.split("/")[1]
            blob_name = "/".join(parsed_url.path.split("/")[2:])
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=self.options.get("azure_credential"),
            )
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            file_content = blob_client.download_blob().readall()
        else:
            with open(file_name, "rb") as file:
                file_content = file.read()

        if self.options.get("base64_encoded") or file_name.endswith(".b64"):
            file_content = base64.b64decode(file_content)

        data = cbor2.loads(file_content)

        records = [data] if isinstance(data, dict) else data

        for record in records:
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected record to be a dict, but got {type(record).__name__}: {record}"
                )
            yield tuple(row for row in convert_to_struct(record, self.schema))
