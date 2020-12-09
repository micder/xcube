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
from typing import Mapping

import xarray as xr

from xcube.core.imgeom import ImageGeom
from xcube.core.rectify import rectify_dataset


def resample_in_space(dataset: xr.Dataset,
                      var_names: Mapping[str, str],
                      output_geom: ImageGeom) -> xr.Dataset:
    """Spatially resample a dataset.

    :param dataset: input dataset to resample
    :param var_names: mapping from variable names to resampling methods
    :param output_geom: output geometry
    :return: dataset resampled to the specified output geometry
    """

    # At present, we only support linear resampling via rectify.
    unsupported_methods = set(var_names.values()) - {"linear"}
    if unsupported_methods:
        raise ValueError("Unsupported resampling method(s): " +
                         ", ".join(unsupported_methods))

    return rectify_dataset(
        dataset=dataset,
        var_names=list(var_names.keys()),
        output_geom=output_geom
    )
