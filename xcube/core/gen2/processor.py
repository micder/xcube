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
from abc import ABC, abstractmethod
from typing import Any, Optional, List

from xcube.constants import EXTENSION_POINT_DATA_PROCESSORS
from xcube.util.assertions import assert_given
from xcube.util.extension import Extension
from xcube.util.extension import ExtensionPredicate
from xcube.util.extension import ExtensionRegistry
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.plugin import get_extension_registry


#######################################################
# Data processor instantiation and registry query
#######################################################

def new_data_processor(processor_id: str,
                       extension_registry: Optional[ExtensionRegistry] = None,
                       **processor_params) -> 'DataProcessor':
    """
    Get an instance of the data processor identified by *processor_id*.

    :param processor_id: The data opener identifier.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :param processor_params: Implementation specific processor parameters.
    :return: A data processor instance.
    """
    assert_given(processor_id, 'processor_id')
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.get_component(EXTENSION_POINT_DATA_PROCESSORS, processor_id)(**processor_params)


def find_data_processor_extensions(predicate: ExtensionPredicate = None,
                                   extension_registry: Optional[ExtensionRegistry] = None) -> List[Extension]:
    """
    Get registered data writer extensions using the optional filter function *predicate*.

    :param predicate: An optional filter function.
    :param extension_registry: Optional extension registry. If not given, the global extension registry will be used.
    :return: List of matching extensions.
    """
    extension_registry = extension_registry or get_extension_registry()
    return extension_registry.find_extensions(EXTENSION_POINT_DATA_PROCESSORS, predicate=predicate)


#######################################################
# Classes
#######################################################


class DataProcessor(ABC):
    """
    An interface that specifies an operation to process a data resource to some arbitrary
    target data resource using arbitrary process parameters.

    Possible process parameters are implementation-specific and are described by a JSON Schema.
    """

    @classmethod
    def get_data_processor_params_schema(cls) -> JsonObjectSchema:
        """
        Get descriptions of parameters that must or can be used to instantiate a new DataProcessor object.
        Parameters are named and described by the properties of the returned JSON object schema.
        The default implementation returns JSON object schema that can have any properties.
        """
        return JsonObjectSchema()

    @classmethod
    @abstractmethod
    def get_process_data_params_schema(cls) -> JsonObjectSchema:
        """
        Get the schema for the parameters passed as *process_params* to :meth:process_data(data, process_params).

        :return: The schema for the allowed parameters in *process_params*.
        :raise DataProcessError: If an error occurs.
        """

    @abstractmethod
    def process_data(self, data: Any, **process_params) -> Any:
        """
        Process a data resource using the supplied *data_id* and *process_params*.

        :param data: The data resource's in-memory representation to be written.
        :param process_params: Processor-specific parameters.
        :return: The schema for the allowed parameters in *process_params*.
        :raise DataProcessError: If an error occurs.
        """


class DataProcessError(Exception):
    """
    Raised on errors in a DataProcessError implementation.

    :param message: The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)
