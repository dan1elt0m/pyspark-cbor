import pytest
from pyspark.sql import SparkSession
import os


# Determine the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
venv_python_path = os.path.join(parent_dir, ".venv/bin/python")

os.environ["PYSPARK_PYTHON"] = venv_python_path

@pytest.fixture(scope="module")
def spark():
    spark_session = (
        SparkSession.builder.appName("SparkCborTest").master("local[*]").getOrCreate()
    )
    yield spark_session
