import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql.types import StructType, StructField, StringType
from src.pyspark_cbor import CBORDataSourceReader

@pytest.fixture
def schema():
    return StructType([StructField("field", StringType(), True)])

@pytest.fixture
def schema():
    return StructType([StructField("field", StringType(), True)])

def test_read_from_s3(schema):
    with patch("boto3.client") as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "key"}]
        }
        mock_s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"mocked content"))}

        reader = CBORDataSourceReader(schema, {"path": "s3://bucket/key"})
        partitions = reader.partitions()
        assert len(partitions) == 1, f"Expected 1 partition, but got {len(partitions)}"


def test_read_from_gcs(schema):
    with patch("google.cloud.storage.Client") as mock_storage_client:
        mock_client = MagicMock()
        mock_storage_client.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_bytes.return_value = b"mocked content"

        reader = CBORDataSourceReader(schema, {"path": "gs://bucket/key"})
        partitions = reader.partitions()
        assert len(partitions) == 1

def test_read_from_azure_blob(schema):
    with patch("azure.storage.blob.BlobServiceClient") as mock_blob_service_client:
        mock_service_client = MagicMock()
        mock_blob_service_client.return_value = mock_service_client
        mock_container_client = MagicMock()
        mock_service_client.get_container_client.return_value = mock_container_client
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob.return_value.readall.return_value = b"mocked content"

        reader = CBORDataSourceReader(schema, {"path": "https://account.blob.core.windows.net/container/blob"})
        partitions = reader.partitions()
        assert len(partitions) == 1