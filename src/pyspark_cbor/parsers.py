"""Parsers for converting Python types from CBOR into PyArrow supported datatypes"""

from typing import Union, Optional, List, Dict, Any

from _cbor2 import CBORSimpleValue, undefined, CBORTag
from pyspark.sql.types import (
    Row,
    DecimalType,
    LongType,
    StructField,
    ArrayType,
    DataType,
    StructType,
    StringType,
    BooleanType,
    DoubleType,
    IntegerType,
    MapType,
)
from decimal import Decimal
import math


def convert_to_struct(data: Dict, schema: Optional[StructType] = None) -> Row:
    # Each field is a dictionary with the field name as the key
    if not schema:
        schema = _infer_schema(data)
    row = {}
    for field in schema.fields:
        output = _parse_field(field.dataType, data.get(field.name))
        row[field.name] = output
    return Row(**row)


def _infer_schema(data: Dict[str, Any]) -> StructType:
    fields = []
    for key, value in data.items():
        if isinstance(value, dict):
            field_type = _infer_schema(value)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                field_type = ArrayType(_infer_schema(value[0]))
            else:
                field_type = ArrayType(StringType())
        elif isinstance(value, str):
            field_type = StringType()
        elif isinstance(value, int):
            field_type = LongType()
        elif isinstance(value, float):
            field_type = DoubleType()
        elif isinstance(value, bool):
            field_type = BooleanType()
        else:
            field_type = StringType()
        fields.append(StructField(key, field_type, True))
    return StructType(fields)


def _parse_field(data_type: DataType, value):
    if isinstance(value, CBORSimpleValue):
        value = value.value
    if isinstance(value, dict) and data_type.typeName() not in ["map", "struct"]:
        schema = _infer_schema(value)
        return convert_to_struct(value, schema)
    if isinstance(data_type, ArrayType):
        return _parse_array(data_type.elementType, value)
    if isinstance(data_type, MapType):
        return _parse_map(data_type, value)
    if isinstance(data_type, DecimalType):
        return _parse_decimal(value)
    elif isinstance(data_type, LongType):
        return _parse_long(value)
    elif isinstance(data_type, IntegerType):
        return _parse_integer(value)
    elif isinstance(data_type, BooleanType):
        return _parse_boolean(value)
    return value


def _parse_map(data_type: MapType, value: Dict) -> Dict:
    map_values = {}
    if isinstance(value, CBORTag):
        return {value.tag: value.value}
    for key, val in value.items():
        if isinstance(key, tuple):
            # This is a tuple key, which is not supported by PyArrow. we will convert it to a string.
            key = str(key)

        if isinstance(key, CBORTag):
            map_values[_parse_field(data_type.keyType, key.tag)] = _parse_field(
                data_type.valueType, key.value
            )
        else:
            map_values[_parse_field(data_type.keyType, key)] = _parse_field(
                data_type.valueType, val
            )
    return map_values


def _parse_boolean(value: bool) -> Optional[bool]:
    if value is undefined:
        return None
    return value


def _parse_array(data_type: DataType, value: List) -> List:
    values = []
    if not value:
        return values
    for i in value:
        if isinstance(i, dict) and data_type.typeName() not in ["map", "struct"]:
            # This is a item that either needs unnesting or is a record itself.
            for key in i.keys():
                if isinstance(key, CBORSimpleValue):
                    values.append(key.value)
                else:
                    values.append(convert_to_struct(i))
        else:
            values.append(_parse_field(data_type, i))

    return values


def _parse_long(value: int) -> Optional[int]:
    long_limit = 9223372036854775807
    if abs(value) > long_limit:
        return None
    return value


def _parse_integer(value: int) -> Optional[int]:
    integer_limit = 2147483647
    if abs(value) > integer_limit:
        return None
    return value


def _parse_decimal(value: Union[Decimal, float]) -> Optional[Decimal]:
    # Handle special float values. nan, -inf and inf are not supported by DecimalType.
    if isinstance(value, float):
        if value in [float("inf"), float("-inf")]:
            return None
        elif math.isnan(value):
            return None
        else:
            return Decimal(str(value))

    return value
