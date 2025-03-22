import math
from datetime import timezone, datetime, timedelta
from decimal import Decimal
from tempfile import TemporaryDirectory
from typing import List
from uuid import UUID

import pytest
import cbor2

from pyspark.sql.types import (
    ArrayType,
    StringType,
    DecimalType,
    StructType,
    StructField,
    LongType,
    BinaryType,
    BooleanType,
    DoubleType,
    IntegerType,
    DataType,
    MapType,
    TimestampType,
    Row,
)

from pyspark_cbor import CBORDataSourceReader, CBORDataSource, CBORInputPartition
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_FILE = os.path.join(current_dir, "examples.cbor.b64")


@pytest.mark.parametrize(
    "field,datatype,expected",
    [
        ("strings", ArrayType(StringType()), ["", "a", "IETF", '"\\', "ü", "水"]),
        (
            "decimals",
            ArrayType(DecimalType(38, 18)),
            [Decimal("14.123"), Decimal("-14.123"), None, None, None],
        ),
        ("bytes", ArrayType(BinaryType()), [b"", b"\x01\x02\x03\x04", b"\xc2\xc2"]),
        ("simples", ArrayType(IntegerType()), [0, 2, 19, 32]),
        ("simple_key", ArrayType(IntegerType()), [99]),
        ("specials", ArrayType(BooleanType()), [False, True, None, None]),
        (
            "tag_as_key",
            ArrayType(MapType(IntegerType(), StringType())),
            [{6007: "notimportant"}],
        ),
        ("tagged", ArrayType(MapType(IntegerType(), StringType())), [{6000: "Hello"}]),
        (
            "timestamps",
            ArrayType(TimestampType()),
            [
                datetime(2013, 3, 21, 20, 4, tzinfo=timezone.utc),
                datetime(2013, 3, 21, 20, 4, 0, 380841, tzinfo=timezone.utc),
                datetime(2013, 3, 21, 22, 4, tzinfo=timezone(timedelta(seconds=7200))),
                datetime(2013, 3, 21, 20, 4, tzinfo=timezone.utc),
                datetime(2013, 3, 21, 20, 4, tzinfo=timezone.utc),
                datetime(2013, 3, 21, 20, 4, 0, 123456, tzinfo=timezone.utc),
                datetime(2013, 3, 21, 20, 4, tzinfo=timezone.utc),
            ],
        ),
        ("tuple_key", ArrayType(MapType(StringType(), StringType())), [{"(2, 1)": ""}]),
        (
            "uuid",
            ArrayType(StringType()),
            [UUID("5eaffac8-b51e-4805-8127-7fdcc7842faf")],
        ),
    ],
)
def test_reader(field: str, datatype: DataType, expected: list):
    schema = StructType([StructField(field, datatype, True)])
    reader = CBORDataSourceReader(schema, {"path": EXAMPLE_FILE})
    data = list(reader.read(CBORInputPartition(EXAMPLE_FILE)))
    if datatype.typeName() == "array":
        assert data[0][0] == expected
    else:
        assert data == expected


@pytest.mark.parametrize(
    "field,datatype,expected",
    [
        ("strings", ArrayType(StringType()), ["", "a", "IETF", '"\\', "ü", "水"]),
        (
            "decimals",
            ArrayType(DecimalType(38, 18)),
            [Decimal("14.123"), Decimal("-14.123"), None, None, None],
        ),
        (
            "integers",
            ArrayType(LongType()),
            [
                0,
                1,
                10,
                23,
                24,
                100,
                1000,
                1000000,
                1000000000000,
                None,
                None,
                None,
                None,
                -1,
                -10,
                -100,
                -1000,
            ],
        ),
        (
            "bytes",
            ArrayType(BinaryType()),
            [bytearray(b""), bytearray(b"\x01\x02\x03\x04"), bytearray(b"\xc2\xc2")],
        ),
        (
            "floats",
            ArrayType(DoubleType()),
            [1.1, 1e300, -4.1, float("inf"), float("nan"), float("-inf")],
        ),
        ("fraction", ArrayType(StringType()), ["2/5"]),
        (
            "integers",
            ArrayType(IntegerType()),
            [
                0,
                1,
                10,
                23,
                24,
                100,
                1000,
                1000000,
                None,
                None,
                None,
                None,
                None,
                -1,
                -10,
                -100,
                -1000,
            ],
        ),
        (
            "ipaddr",
            ArrayType(StringType()),
            ["192.10.10.1", "2001:db8:85a3::8a2e:370:7334"],
        ),
        (
            "ipnet",
            ArrayType(StringType()),
            ["192.168.0.0/24", "2001:db8:85a3::8a2e:0:0/96"],
        ),
        (
            "regex",
            ArrayType(StringType()),
            ["re.compile('hello (world)')"],
        ),  # not sure how to handle this. Can I do something with a flag to return the compiled regex?
        ("simples", ArrayType(IntegerType()), [0, 2, 19, 32]),
        ("simple_key", ArrayType(IntegerType()), [99]),
        ("specials", ArrayType(BooleanType()), [False, True, None, None]),
        (
            "tag_as_key",
            ArrayType(MapType(IntegerType(), StringType())),
            [{6007: "notimportant"}],
        ),
        ("tagged", ArrayType(MapType(IntegerType(), StringType())), [{6000: "Hello"}]),
        ("uuid", ArrayType(StringType()), ["5eaffac8-b51e-4805-8127-7fdcc7842faf"]),
    ],
)
def test_spark_reader_single_row(spark, field: str, datatype: DataType, expected: List):
    spark.dataSource.register(CBORDataSource)
    schema = StructType(
        [
            StructField(field, datatype, True),
        ]
    )
    df = (
        spark.read.format("cbor")
        .schema(schema)
        .option("base64_encoded", True)
        .load(EXAMPLE_FILE)
    )
    assert df.count() == 1
    assert df.schema.fields[0].name == field
    assert df.schema.fields[0].dataType == datatype
    row = df.collect()
    for i, value in enumerate(row[0][0]):
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(expected[i])
        else:
            assert value == expected[i]


def test_spark_multi_row_multi_part(spark):
    spark.dataSource.register(CBORDataSource)
    # Some example multi row relational data with complex types
    row1 = {
        "name": "Alice",
        "age": 25,
        "city": "San Francisco",
        "address": {"street": "123 Main St", "zipcode": "94105"},
        "attributes": {"height": 5.5, "weight": 130},
        "family": [
            {"name": "Bob", "relation": "brother", "age": 30},
            {"name": "Charlie", "relation": "brother", "age": 35},
        ],
        "previous_addresses": [
            [{"street": "456 Broadway", "zipcode": "10001"}],
            [{"street": "789 Sunset Blvd", "zipcode": "90028"}],
        ],
    }
    row2 = {
        "name": "Bob",
        "age": 30,
        "city": "New York",
        "address": {"street": "456 Broadway", "zipcode": "10001"},
        "attributes": {"height": 6.0, "weight": 180},
        "family": [
            {"name": "Alice", "relation": "sister", "age": 25},
            {"name": "Charlie", "relation": "brother", "age": 35},
        ],
    }
    row3 = {
        "name": "Charlie",
        "age": 35,
        "city": "Los Angeles",
        "address": {"street": "789 Sunset Blvd", "zipcode": "90028"},
        "attributes": {"height": 5.8, "weight": 160},
        "family": [],
    }

    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("age", LongType(), True),
            StructField("city", StringType(), True),
            StructField(
                "address",
                StructType(
                    [
                        StructField("street", StringType(), True),
                        StructField("zipcode", StringType(), True),
                    ]
                ),
                True,
            ),
            StructField("attributes", MapType(StringType(), StringType()), True),
            StructField(
                "family",
                ArrayType(
                    StructType(
                        [
                            StructField("name", StringType(), True),
                            StructField("relation", StringType(), True),
                            StructField("age", LongType(), True),
                        ]
                    )
                ),
                True,
            ),
            StructField(
                "previous_addresses",
                ArrayType(
                    ArrayType(
                        StructType(
                            [
                                StructField("street", StringType(), True),
                                StructField("zipcode", StringType(), True),
                            ]
                        )
                    )
                ),
                True,
            ),
        ]
    )

    data1 = [row1, row2]
    data2 = [row3]
    with TemporaryDirectory() as tempdir:
        tempfile = f"{tempdir}/example1.cbor"
        with open(tempfile, "wb") as file:
            cbor_data1 = cbor2.dumps(data1)
            file.write(cbor_data1)

        tempfile2 = f"{tempdir}/example2.cbor"
        with open(tempfile2, "wb") as file:
            cbor_data2 = cbor2.dumps(data2)
            file.write(cbor_data2)

        df = spark.read.format("cbor").schema(schema).load(tempdir)
        assert sorted(df.collect()) == sorted(
            [
                Row(
                    name="Alice",
                    age=25,
                    city="San Francisco",
                    address=Row(street="123 Main St", zipcode="94105"),
                    attributes={"weight": "130", "height": "5.5"},
                    family=[
                        Row(name="Bob", relation="brother", age=30),
                        Row(name="Charlie", relation="brother", age=35),
                    ],
                    previous_addresses=[
                        [Row(street="456 Broadway", zipcode="10001")],
                        [Row(street="789 Sunset Blvd", zipcode="90028")],
                    ],
                ),
                Row(
                    name="Bob",
                    age=30,
                    city="New York",
                    address=Row(street="456 Broadway", zipcode="10001"),
                    attributes={"weight": "180", "height": "6.0"},
                    family=[
                        Row(name="Alice", relation="sister", age=25),
                        Row(name="Charlie", relation="brother", age=35),
                    ],
                    previous_addresses=[],
                ),
                Row(
                    name="Charlie",
                    age=35,
                    city="Los Angeles",
                    address=Row(street="789 Sunset Blvd", zipcode="90028"),
                    attributes={"weight": "160", "height": "5.8"},
                    family=[],
                    previous_addresses=[],
                ),
            ]
        )
