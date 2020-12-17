# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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
from typing import Sequence, Dict, Any

import click

from xcube.constants import FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM
from xcube.core.imgeom import ImageGeom

UPSAMPLING_METHODS = ['asfreq', 'ffill', 'bfill', 'pad', 'nearest', 'interpolate']
DOWNSAMPLING_METHODS = ['count', 'first', 'last', 'min', 'max', 'sum', 'prod', 'mean', 'median', 'std', 'var']
RESAMPLING_METHODS = UPSAMPLING_METHODS + DOWNSAMPLING_METHODS

SPLINE_INTERPOLATION_KINDS = ['zero', 'slinear', 'quadratic', 'cubic']
OTHER_INTERPOLATION_KINDS = ['linear', 'nearest', 'previous', 'next']
INTERPOLATION_KINDS = SPLINE_INTERPOLATION_KINDS + OTHER_INTERPOLATION_KINDS

OUTPUT_FORMAT_NAMES = [FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4, FORMAT_NAME_MEM]

DEFAULT_OUTPUT_PATH = 'out.zarr'
DEFAULT_RESAMPLING_METHOD = 'mean'
DEFAULT_RESAMPLING_FREQUENCY = '1D'
DEFAULT_RESAMPLING_BASE = 0
DEFAULT_INTERPOLATION_KIND = 'linear'


# noinspection PyShadowingBuiltins
@click.command(name='resample')
@click.argument('cube')
@click.option('--config', '-c', metavar='CONFIG', multiple=True,
              help='xcube dataset configuration file in YAML format. More than one config input file is allowed.'
                   'When passing several config files, they are merged considering the order passed via command line.')
@click.option('--output', '-o', metavar='OUTPUT',
              default=DEFAULT_OUTPUT_PATH,
              help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}.")
@click.option('--format', '-f',
              type=click.Choice(OUTPUT_FORMAT_NAMES),
              help="Output format. If omitted, format will be guessed from output path.")
@click.option('--variables', '--vars', metavar='VARIABLES',
              help="Comma-separated list of names of variables to be included.")
@click.option('--method', '-M', multiple=True,
              help=f"Temporal resampling method. "
                   f"Available downsampling methods are "
                   f"{', '.join(map(repr, DOWNSAMPLING_METHODS))}, "
                   f"the upsampling methods are "
                   f"{', '.join(map(repr, UPSAMPLING_METHODS))}. "
                   f"If the upsampling method is 'interpolate', "
                   f"the option '--kind' will be used, if given. "
                   f"Other upsampling methods that select existing values "
                   f"honour the '--tolerance' option. "
                   f'Defaults to {DEFAULT_RESAMPLING_METHOD!r}.')
@click.option('--frequency', '-F',
              help='Temporal aggregation frequency. Use format "<count><offset>" '
                   "where <offset> is one of 'H', 'D', 'W', 'M', 'Q', 'Y'. "
                   "Use 'all' to aggregate all time steps included in the dataset." 
                   f'Defaults to {DEFAULT_RESAMPLING_FREQUENCY!r}.')
@click.option('--offset', '-O',
              help='Offset used to adjust the resampled time labels. Uses same syntax as frequency. '
                   'Some Pandas date offset strings are supported as well.')
@click.option('--base', '-B', type=int, default=DEFAULT_RESAMPLING_BASE,
              help='For frequencies that evenly subdivide 1 day, the origin of the aggregated intervals. '
                   "For example, for '24H' frequency, base could range from 0 through 23. "
                   f'Defaults to {DEFAULT_RESAMPLING_BASE!r}.')
@click.option('--kind', '-K', type=str, default=DEFAULT_INTERPOLATION_KIND,
              help="Interpolation kind which will be used if upsampling method is 'interpolation'. "
                   f"May be one of {', '.join(map(repr, INTERPOLATION_KINDS))} where "
                   f"{', '.join(map(repr, SPLINE_INTERPOLATION_KINDS))} refer to a spline interpolation of "
                   f"zeroth, first, second or third order; 'previous' and 'next' "
                   f"simply return the previous or next value of the point. "
                   "For more info "
                   "refer to scipy.interpolate.interp1d(). "
                   f'Defaults to {DEFAULT_INTERPOLATION_KIND!r}.')
@click.option('--tolerance', '-T', type=str,
              help='Tolerance for selective upsampling methods. Uses same syntax as frequency. '
                   'If the time delta exceeds the tolerance, '
                   'fill values (NaN) will be used. '
                   'Defaults to the given frequency.')
@click.option("--xmin", type=float,
              help="x minimum for spatial resampling")
@click.option("--ymin", type=float,
              help="y minimum for spatial resampling")
@click.option("--xsize", type=float,
              help="y size for spatial resampling")
@click.option("--ysize", type=float,
              help="y size for spatial resampling")
@click.option("--resolution", type=float,
              help="x and y resolution for spatial resampling")
@click.option("--coregister-to", "-C", type=str,
              help="Filename of a cube to use as a coregistration target")
@click.option("--spatial-first", "-s", default=False, is_flag=True,
              help="Do the spatial resampling first when performing both "
                   "spatial and temporal resampling. If this option is "
                   "omitted, temporal resampling will be done first.")
@click.option('--dry-run', default=False, is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
def resample(cube,
             config,
             output,
             format,
             variables,
             method,
             frequency,
             offset,
             base,
             kind,
             tolerance,
             xmin,
             ymin,
             xsize,
             ysize,
             resolution,
             coregister_to,
             spatial_first,
             dry_run):
    """
    Resample data along the time dimension.
    """

    input_path = cube
    config_files = config
    output_path = output
    output_format = format

    from xcube.util.config import load_configs
    config = load_configs(*config_files) if config_files else {}

    if input_path:
        config['input_path'] = input_path
    if output_path:
        config['output_path'] = output_path
    if output_format:
        config['output_format'] = output_format
    if method:
        config['methods'] = method
    if frequency:
        config['frequency'] = frequency
    if offset:
        config['offset'] = offset
    if offset:
        config['base'] = base
    if kind:
        config['interp_kind'] = kind
    if tolerance:
        config['tolerance'] = tolerance
    if xmin:
        config["xmin"] = xmin
    if ymin:
        config["ymin"] = ymin
    if xsize:
        config["xsize"] = xsize
    if ysize:
        config["ysize"] = ysize
    if resolution:
        config["resolution"] = resolution
    if coregister_to:
        config["coregister_to"] = coregister_to
    if spatial_first:
        config["spatial_first"] = spatial_first
    if variables:
        try:
            variables = set(map(lambda c: str(c).strip(), variables.split(',')))
        except ValueError:
            variables = None
        if variables is not None \
                and next(iter(True for var_name in variables if var_name == ''), False):
            variables = None
        if variables is None or len(variables) == 0:
            raise click.ClickException(f'invalid variables {variables!r}')
        config['variables'] = variables

    if 'methods' in config:
        methods = config['methods']
        for method in methods:
            if method not in RESAMPLING_METHODS:
                raise click.ClickException(f'invalid resampling method {method!r}')

    # noinspection PyBroadException
    _resample(**config, dry_run=dry_run, monitor=print)

    return 0


def _resample(input_path: str = None,
              variables: Sequence[str] = None,
              metadata: Dict[str, Any] = None,
              output_path: str = DEFAULT_OUTPUT_PATH,
              output_format: str = None,
              methods: Sequence[str] = (DEFAULT_RESAMPLING_METHOD,),
              frequency: str = DEFAULT_RESAMPLING_FREQUENCY,
              offset: str = None,
              base: int = DEFAULT_RESAMPLING_BASE,
              interp_kind: str = DEFAULT_INTERPOLATION_KIND,
              tolerance: str = None,
              xmin: float = None,
              ymin: float = None,
              xsize: float = None,
              ysize: float = None,
              resolution: float = None,
              coregister_to: str = None,
              spatial_first: bool = False,
              dry_run: bool = False,
              monitor=None):
    from xcube.core.dsio import guess_dataset_format
    from xcube.core.dsio import open_cube
    from xcube.core.dsio import write_cube
    from xcube.core.resample import resample_in_time
    from xcube.core.resample_spatial import resample_in_space
    from xcube.core.update import update_dataset_chunk_encoding

    if not output_format:
        output_format = guess_dataset_format(output_path)

    if coregister_to:
        monitor(f"Opening coregistration target from {coregister_to}...")
        coregistration_target = open_cube(coregister_to)
        monitor("Done.")
    else:
        coregistration_target = None

    monitor(f'Opening cube from {input_path!r}...')
    with open_cube(input_path) as ds:

        monitor('Resampling...')

        def resample_time_with_params(input_ds):
            return resample_in_time(input_ds,
                                    frequency=frequency,
                                    method=methods,
                                    offset=offset,
                                    base=base,
                                    interp_kind=interp_kind,
                                    tolerance=tolerance,
                                    time_chunk_size=1,
                                    var_names=variables,
                                    metadata=metadata)

        def resample_space_with_params(input_ds):
            if coregistration_target:
                return resample_in_space(input_ds,
                                         var_names=variables,
                                         coregister_to=coregistration_target)
            else:
                output_geom = ImageGeom(size=(int(xsize), int(ysize)),
                                        tile_size=None,
                                        x_min=xmin,
                                        y_min=ymin,
                                        xy_res=resolution,
                                        is_geo_crs=False)
                return resample_in_space(input_ds,
                                         var_names=variables,
                                         output_geom=output_geom)

        if spatial_first:
            agg_ds = resample_time_with_params(resample_space_with_params(ds))
        else:
            agg_ds = resample_space_with_params(resample_time_with_params(ds))

        agg_ds = update_dataset_chunk_encoding(agg_ds,
                                               chunk_sizes={},
                                               format_name=output_format,
                                               in_place=True)

        monitor(f'Writing resampled cube to {output_path!r}...')
        if not dry_run:
            write_cube(agg_ds, output_path, output_format, cube_asserted=True)

        monitor(f'Done.')
