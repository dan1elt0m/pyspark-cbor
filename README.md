# pyspark--cbor

> By no means ready for production use. This is still in development and now just see it as an example.

This repository showcases custom Spark data source `cbor` built using the new [**Python Data Source API**](https://issues.apache.org/jira/browse/SPARK-44076) for the upcoming Apache Spark 4.0 release.
For an in-depth understanding of the API, please refer to the [API source code](https://github.com/apache/spark/blob/master/python/pyspark/sql/datasource.py).


```python
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CBOR Data Source Example") \
    .getOrCreate()

# Register the CBOR data source
spark.dataSource.register(CBORDataSource)

# Read CBOR file
df = spark.read.format("cbor").load("path/to/your/file.cbor")

# Show the DataFrame
df.show()

```

### Options