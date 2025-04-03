# pyspark-cbor

> This project is still in development. See the [TODOs](#todos) section for more information. If you find any issues, please let me know.

This library implements custom Spark data source `cbor` built using the new [**Python Data Source API**](https://issues.apache.org/jira/browse/SPARK-44076) for the upcoming Apache Spark 4.0 release.
For an in-depth understanding of the API, please refer to the [API source code](https://github.com/apache/spark/blob/master/python/pyspark/sql/datasource.py).

### Supported features
- Support all CBOR data types, including nested structures. Caveat: CBOR is more flexible than Spark's schema, so some data may be lost. See Permissive mode section for more information.
- Read CBOR file(s) (in parallel) with a specified schema
- Read CBOR file(s) with base64 encoded values
- Read CBOR file(s) from a local filepath or azure, aws or gcp storage 

### Installation
```bash
pip install pyspark-cbor
```

This library assumes that you have already installed Apache Spark and PySpark. 
If not, you can install it using the following command:

```bash
pip install pyspark-cbor[spark]
```

### Example

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark_cbor import CBORDataSource

# Initialize Spark session
spark = SparkSession.builder
.appName("CBOR Data Source Example")
.getOrCreate()

# Register the CBOR data source
spark.dataSource.register(CBORDataSource)

schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("city", StringType(), True)
])

# Read CBOR file
df = spark.read.format("cbor").schema(schema).load("path/to/your/file.cbor")

# Show the DataFrame
df.show()

```

### Settings 
- `path`: The path to the CBOR file or directory containing CBOR files.
- `schema`: The schema of the CBOR file. Currently, the schema can't be specified as string
- `mode`: The mode of reading the CBOR file. The default mode is `PERMISSIVE`. 

### Options
- `base64_decoded` (default: `False`): Whether to decode base64 encoded values before parsing with CBOR. Will always decode if file ends with `.b64`

### Limitations
- Extensive unit tests, but not yet used anywhere except on local machine
- Nested structures are recursively parsed.
  This means that the maximum depth of the nested structure is limited by the maximum recursion depth of Python.
- Not sure if I implemented the `CBOR TAGS` and other special types correctly. Might or might not work as expected.
- Not all CBOR data can be represented in a Spark DataFrame, see Permissive Mode.

### Permissive mode
Spark converts CBOR data into an Arrow DataFrame based on the provided schema.
Not all CBOR data can be represented in the schema.  
This library tries to preserve data as much as possible, even if the schema doesn't match. 
However, the following happens in permissive mode:
- If a field is undefined, it will be set to `null`
- If a field is defined but the value is not present, it will be set to `null`.

Integers will be set to null if they exceed the maximum value of the corresponding Spark type:
   - IntegerType: `2147483647`
   - LongType: `9223372036854775807`

- In DecimalType, the precision is limited to 38 digits. infinity and NaN are not supported and will be converted to `null`.

### TODOs
- Add StreamReader 
- Add Writer
- Add StreamWriter
- Add more options, such as `RESTRICTIVE` mode
- Add more examples
- Add more documentation
- Add more error handling
- Support string ddl schema specification
- Add more logging
- Add more performance optimizations: e.g., can file splitting be done?

### Contributing
Feel free to contribute to this project. As you can see there is still a lot of work to be done.
