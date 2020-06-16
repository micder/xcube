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
from typing import Any, Sequence

from xcube.core.gen2.request import InputConfig, OutputConfig


class DataGenerator(ABC):
    """
    An interface that specifies an operation to process a data resource to some arbitrary
    target data resource using arbitrary process parameters.

    Possible process parameters are implementation-specific and are described by a JSON Schema.
    """

    @classmethod
    @abstractmethod
    def get_gen_data_params_schema(cls):
        """
        Get the schema for the parameters passed as *process_params* to :meth:process_data(data, process_params).

        :return: The schema for the allowed parameters in *process_params*.
        :raise DataProcessError: If an error occurs.
        """

    @abstractmethod
    def gen_data(self,
                 input_configs: Sequence[InputConfig],
                 output_configs: Sequence[OutputConfig],
                 **gen_params) -> Any:
        """
        Generate a data resource using the supplied *data_id* and *process_params*.

        :param input_configs: Input configurations.
        :param output_configs: Output configurations.
        :param gen_params: Generate parameters.
        :return: The schema for the allowed parameters in *process_params*.
        :raise DataProcessError: If an error occurs.
        """


class CombineInputsDataGenerator(DataGenerator, ABC):
    """
    Base class for a generator that uses multiple inputs and combines
    them into a single data resource which is then written to one or more outputs.
    """

    @classmethod
    @abstractmethod
    def get_gen_data_params_schema(cls):
        """
        Get the schema for the parameters passed as *process_params* to :meth:process_data(data, process_params).

        :return: The schema for the allowed parameters in *process_params*.
        :raise DataProcessError: If an error occurs.
        """

    @abstractmethod
    def gen_data(self,
                 input_configs: Sequence[InputConfig],
                 output_configs: Sequence[OutputConfig],
                 **gen_params) -> Any:
        """
        Generate a data resource using the supplied *data_id* and *process_params*.

        :param input_configs: Input configurations.
        :param output_configs: Output configurations.
        :param gen_params: Generate parameters.
        :return: The schema for the allowed parameters in *process_params*.
        :raise DataProcessError: If an error occurs.
        """
