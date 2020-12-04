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

import pytest
import numpy as np

from xcube.core.imgeom import ImageGeom
from xcube.core.new import new_cube
from xcube.core.resample_spatial import resample_in_space


def scale_up(array, scale):
    return array.repeat(scale, axis=1).repeat(scale, axis=2)


def array_to_cube(array):
    width = array.shape[1]
    height = array.shape[2]
    return new_cube(width=width,
                    height=height,
                    time_periods=1,
                    x_start=0, y_start=0,
                    x_res=4 / width,
                    y_res=4 / height,
                    variables={"var1": array})


def diagnostic_plot(input_cube, expected, actual, scale):
    import matplotlib.pyplot as plt
    from pathlib import Path

    def add_plot(subplot, cube, name):
        plt.subplot(1, 3, subplot)
        cube.var1.plot(edgecolors="gray", vmin=-1, vmax=2)
        plt.title(name)

    plt.rcParams["figure.figsize"] = (14, 4)
    add_plot(1, input_cube, "input")
    add_plot(2, expected, "expected")
    add_plot(3, actual, "actual")
    plt.subplots_adjust(left=0.05, right=0.95)
    home = str(Path.home())
    plt.savefig(f"{home}/rectify-{scale}.png")


@pytest.mark.xfail
@pytest.mark.parametrize("scale_factor", [2])
def test_basic_pattern(scale_factor, diagnostics_option):
    pattern = np.array([[
        [2, 2, 1, 1],
        [2, 2, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]])

    input_cube = array_to_cube(pattern)
    expected_output = array_to_cube(scale_up(pattern, scale_factor))
    geometry = ImageGeom(
        (expected_output.dims["lon"], expected_output.dims["lat"]),
        xy_res=1 / scale_factor
    )
    actual_output = resample_in_space(
        dataset=input_cube,
        var_names="var1",
        output_geom=geometry
    )

    if diagnostics_option:
        diagnostic_plot(input_cube, expected_output, actual_output,
                        scale_factor)

    np.testing.assert_equal(
        expected_output.var1.data,
        actual_output.var1.data
    )
