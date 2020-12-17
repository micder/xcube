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
from typing import Sequence
from typing import Union

import xarray as xr

from xcube.core.imgeom import ImageGeom
from xcube.core.rectify import rectify_dataset


def resample_in_space(dataset: xr.Dataset,
                      var_names: Union[Mapping[str, str], Sequence[str]],
                      output_geom: ImageGeom = None,
                      coregister_to: xr.Dataset = None,
                      ) -> xr.Dataset:
    """Spatially resample a dataset.

    :param dataset: input dataset to resample
    :param var_names: mapping from variable names to resampling method names,
           or sequence of variable names (in which case the default method will
           be used for all of them)
    :param output_geom: output geometry
    :param coregister_to: coregister `dataset` to this dataset
    :return: dataset resampled to the specified output geometry
    """

    if isinstance(var_names, list):
        # Default: set linear resampling for all variables
        var_names = {var_name: "linear" for var_name in var_names}
    else:
        # At present, we only support linear resampling via rectify.
        unsupported_methods = set(var_names.values()) - {"linear"}
        if unsupported_methods:
            raise ValueError("Unsupported resampling method(s): " +
                             ", ".join(unsupported_methods))

    # either the output geometry or a coregistration target must be specified
    if (output_geom is None) == (coregister_to is None):
        raise ValueError("Precisely one of output_geom and coregister_to "
                         "must be supplied.")

    if output_geom is None:
        output_geom = ImageGeom.from_dataset(coregister_to)

    return rectify_dataset(
        dataset=dataset,
        var_names=list(var_names.keys()),
        output_geom=output_geom
    )
