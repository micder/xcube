# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os.path
import sys
from typing import Optional, Type, Dict, Any, Sequence, Mapping

import yaml

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_given
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


class InputConfig:
    def __init__(self,
                 store_id: str = None,
                 opener_id: str = None,
                 processor_id: str = None,
                 data_id: str = None,
                 variable_names: Sequence[str] = None,
                 store_params: Mapping[str, Any] = None,
                 open_params: Mapping[str, Any] = None,
                 process_params: Mapping[str, Any] = None):
        assert_condition(store_id or opener_id, 'One of store_id and opener_id must be given')
        assert_condition(not process_params or process_params and processor_id, 'processor_id must be given')
        assert_given(data_id, 'data_id')
        self.store_id = store_id
        self.opener_id = opener_id
        self.processor_id = processor_id
        self.data_id = data_id
        self.variable_names = variable_names
        self.store_params = store_params or {}
        self.open_params = open_params or {}
        self.process_params = open_params or {}

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                store_id=JsonStringSchema(min_length=1),
                opener_id=JsonStringSchema(min_length=1),
                data_id=JsonStringSchema(min_length=1),
                variable_names=JsonArraySchema(items=JsonStringSchema(min_length=1), min_items=1),
                store_params=JsonObjectSchema(),
                open_params=JsonObjectSchema()
            ),
            additional_properties=False,
            required=['data_id'],
            factory=cls,
        )


class OutputConfig:

    def __init__(self,
                 store_id: str = None,
                 writer_id: str = None,
                 processor_id: str = None,
                 data_id: str = None,
                 store_params: Mapping[str, Any] = None,
                 write_params: Mapping[str, Any] = None,
                 process_params: Mapping[str, Any] = None):
        assert_condition(store_id or writer_id, 'One of store_id and writer_id must be given')
        assert_condition(not process_params or process_params and processor_id, 'processor_id must be given')
        self.store_id = store_id
        self.writer_id = writer_id
        self.processor_id = processor_id
        self.data_id = data_id
        self.store_params = store_params or {}
        self.write_params = write_params or {}
        self.process_params = process_params or {}

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                store_id=JsonStringSchema(min_length=1),
                writer_id=JsonStringSchema(min_length=1),
                data_id=JsonStringSchema(default=None),
                store_params=JsonObjectSchema(),
                write_params=JsonObjectSchema(),
            ),
            additional_properties=False,
            required=[],
            factory=cls,
        )


class GenConfig:

    def __init__(self,
                 gen_params: Mapping[str, Any] = None):
        self.gen_params = gen_params or {}

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(factory=cls)


class Request:
    def __init__(self,
                 input_configs: Sequence[InputConfig] = None,
                 gen_configs: GenConfig = None,
                 output_config: OutputConfig = None):
        assert_given(input_configs, 'input_configs')
        assert_given(output_config, 'output_config')
        self.input_configs = input_configs
        self.gen_configs = gen_configs
        self.output_config = output_config

    @classmethod
    def get_schema(cls):
        return JsonObjectSchema(
            properties=dict(
                input_configs=JsonArraySchema(items=InputConfig.get_schema(), min_items=1),
                gen_config=GenConfig.get_schema(),
                output_config=OutputConfig.get_schema(),
            ),
            required=['input_configs', 'output_config'],
            factory=cls,
        )

    def to_dict(self) -> Mapping[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        return self.get_schema().to_instance(self)

    @classmethod
    def from_dict(cls, request_dict: Dict) -> 'Request':
        """Create new instance from a JSON-serializable dictionary"""
        return cls.get_schema().from_instance(request_dict)

    @classmethod
    def from_file(cls, request_file: Optional[str], exception_type: Type[BaseException] = ValueError) -> 'Request':
        """Create new instance from a JSON file, or YAML file, or JSON passed via stdin."""
        request_dict = cls._load_request_file(request_file, exception_type=exception_type)
        return cls.from_dict(request_dict)

    @classmethod
    def _load_request_file(cls, request_file: Optional[str], exception_type: Type[BaseException] = ValueError) -> Dict:

        if request_file is not None and not os.path.exists(request_file):
            raise exception_type(f'Generator request "{request_file}" not found.')

        try:
            if request_file is None:
                if not sys.stdin.isatty():
                    return json.load(sys.stdin)
            else:
                with open(request_file, 'r') as fp:
                    if request_file.endswith('.json'):
                        return json.load(fp)
                    else:
                        return yaml.safe_load(fp)
        except BaseException as e:
            raise exception_type(f'Error loading generator request "{request_file}": {e}') from e

        raise exception_type(f'Missing generator request.')
