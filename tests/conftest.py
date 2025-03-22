import pytest
from pyspark.sql import SparkSession
import os

os.environ['PYSPARK_PYTHON'] = "../.venv/bin/python"


@pytest.fixture(scope="module")
def spark():
    spark_session = (
        SparkSession.builder.appName("SparkCborTest")
        .master("local[*]").getOrCreate()
    )
    yield spark_session
