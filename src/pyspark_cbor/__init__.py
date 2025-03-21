import base64
from pyspark.sql.datasource import InputPartition, DataSource, DataSourceReader
from pyspark.sql.types import StructType
from typing import Iterator, Tuple
import cbor2
from src.pyspark_cbor.parsers import _parse_array, _parse_record


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



    def read(self, partition: InputPartition) -> Iterator[Tuple]:
        # Implement the logic to read CBOR data
        file_path = self.options.get("path")
        base64_encoded = self.options.get("base64Encoded", False)
        with open(file_path, "rb") as file:
            file_content = file.read()
            if base64_encoded:
                file_content = base64.b64decode(file_content)
            data = cbor2.loads(file_content)

        records = [data] if isinstance(data, dict) else data

        for record in records:
            yield tuple(row for row in _parse_record(self.schema, record))
