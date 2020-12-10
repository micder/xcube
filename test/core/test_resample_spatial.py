#  The MIT License (MIT)
#
#  Copyright (c) 2020 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
#

# Uncomment the following to disable numba JIT compilation for debugging:
# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import pytest
import numpy as np

from xcube.core.imgeom import ImageGeom
from xcube.core.new import new_cube
from xcube.core.resample_spatial import resample_in_space


@pytest.fixture
def diagnostics_option(request):
    """Return the value of the custom command-line flag --diagnosticsresample

    The flag indicates whether `test_basic_pattern` should write diagnostic
    images to the user's home directory for debugging purposes.

    The flag is defined in `conftest.py` in the root `tests` directory.
    (Command-line options may only be defined in the root directory: see
    https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_addoption .)

    :param request: the request object
    :return: the value of the `--diagnosticsresample` flag
    """
    return request.config.getoption("--diagnosticsresample")


def scale_up(array, scale):
    """Scale a NumPy array up by the given factor

    Scaling is done by reduplicating values in axes 1 and 2 (it is assumed
    that axis 1 is time).

    To conform to the current behaviour of `rectify_dataset`, the edge of the
    scaled-up array is overwritten with a NaN-valued border of a width
    corresponding to half a source pixel (scale // 2).

    :param array: input array
    :param scale: scale factor (positive integer)
    :return: scaled array
    """
    scaled = array.repeat(scale, axis=1).repeat(scale, axis=2)
    border_width = scale // 2
    if border_width > 0:
        fill_value = -(2 ** 63)  # NaN
        scaled[0, 0:border_width] = fill_value
        scaled[0, -border_width:] = fill_value
        scaled[0, :, 0:border_width] = fill_value
        scaled[0, :, -border_width:] = fill_value
    return scaled


def array_to_cube(array):
    """Make a data cube containing the given array as "var1"

    :param array: data array
    :return: cube containing data from array
    """
    width = array.shape[1]
    height = array.shape[2]
    return new_cube(width=width,
                    height=height,
                    time_periods=1,
                    x_start=0, y_start=0,
                    x_res=4 / width,
                    y_res=4 / height,
                    variables={"var1": array})


@pytest.fixture
def minimal_cube():
    return new_cube(width=2, height=2, time_periods=1,
                    x_start=0, y_start=0, x_res=2, y_res=2,
                    variables={"var1": np.array([[[1, 2], [3, 4]]])})


def diagnostic_plot(input_cube, expected, actual, scale):
    """Save images of data arrays to user's home directory

    Only intended for diagnostic and debugging purposes, not execution during
    regular unit testing.

    :param input_cube: input data cube
    :param expected: expected output data cube
    :param actual: actual output data cube
    :param scale: scale factor (used for filename)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    def add_plot(subplot, cube, name):
        plt.subplot(1, 3, subplot)
        cube.var1.plot(edgecolors="gray", vmin=-1, vmax=16)
        plt.title(name)

    plt.rcParams["figure.figsize"] = (14, 4)
    plt.clf()
    add_plot(1, input_cube, "input")
    add_plot(2, expected, "expected")
    add_plot(3, actual, "actual")
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.savefig(f"{str(Path.home())}/rectify-{scale}.png")


@pytest.mark.parametrize("pattern", [
    [[2, 2, 1, 1],
     [2, 2, 0, 1],
     [1, 0, 0, 1],
     [1, 1, 1, 1]],
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]],
])
@pytest.mark.parametrize("scale_factor", [1, 2, 3, 4])
# Not currently testing coregistration, since it's buggy.
@pytest.mark.parametrize("coregister", [False])
def test_basic_pattern(pattern, scale_factor, coregister, diagnostics_option):
    """Test that a small data pattern is correctly resampled to a higher
    resolution.

    :param pattern: two-dimensional data pattern
    :param scale_factor: factor by which the resolution should be scaled
           (positive integer)
    :param coregister: test `coregister_to` rather than `output_geom`
    :param diagnostics_option:
           True if images should be saved to the home directory for
           diagnostic/debugging purposes
    """
    pattern = np.array([pattern])
    input_cube = array_to_cube(pattern)
    expected_output = array_to_cube(scale_up(pattern, scale_factor))
    geometry = ImageGeom(
        (expected_output.dims["lon"], expected_output.dims["lat"]),
        xy_res=1 / scale_factor
    )
    # geom_from_dataset = ImageGeom.from_dataset(expected_output)
    arguments = dict(
        dataset=input_cube,
        var_names={"var1": "linear"},
    )
    if coregister:
        arguments["coregister_to"] = expected_output
    else:
        arguments["output_geom"] = geometry

    actual_output = resample_in_space(**arguments)

    if diagnostics_option:
        diagnostic_plot(input_cube, expected_output, actual_output,
                        scale_factor)

    np.testing.assert_equal(
        expected_output.var1.data,
        actual_output.var1.data
    )


def test_unknown_resampling_methods(minimal_cube):
    with pytest.raises(ValueError) as exception_info:
        resample_in_space(minimal_cube,
                          var_names={"var1": "an_unknown_method"},
                          output_geom=ImageGeom(2, 2, 1)
                          )
        assert "an_unknown_method" in str(exception_info.value)


def test_neither_geometry_nor_coregistration_given(minimal_cube):
    with pytest.raises(ValueError) as _:
        resample_in_space(minimal_cube,
                          var_names={"var1": "an_unknown_method"},
                          )
