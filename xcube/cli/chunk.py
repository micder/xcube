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

import click

from xcube.util.cliutil import parse_cli_kwargs

DEFAULT_OUTPUT_PATH = 'out.zarr'


# noinspection PyShadowingBuiltins
@click.command(name="chunk")
@click.argument('cube')
@click.option('--output', '-o', metavar='OUTPUT', default=DEFAULT_OUTPUT_PATH,
              help=f'Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}')
@click.option('--format', '-f', metavar='FORMAT', type=click.Choice(['zarr', 'netcdf']),
              help="Format of the output. If not given, guessed from OUTPUT.")
@click.option('--params', '-p', metavar='PARAMS',
              help="Parameters specific for the output format."
                   " Comma-separated list of <key>=<value> pairs.")
@click.option('--chunks', '-C', metavar='CHUNKS', nargs=1, default=None,
              help='Chunk sizes for each dimension.'
                   ' Comma-separated list of <dim>=<size> pairs,'
                   ' e.g. "time=1,lat=270,lon=270"')
def chunk(cube, output, format=None, params=None, chunks=None):
    """
    (Re-)chunk xcube dataset.
    Changes the external chunking of all variables of CUBE according to CHUNKS and writes
    the result to OUTPUT.
    """
    chunk_sizes = None
    if chunks:
        chunk_sizes = parse_cli_kwargs(chunks, metavar="CHUNKS")
        for k, v in chunk_sizes.items():
            if not isinstance(v, int) or v <= 0:
                raise click.ClickException("Invalid value for CHUNKS, "
                                           f"chunk sizes must be positive integers: {chunks}")

    write_kwargs = dict()
    if params:
        write_kwargs = parse_cli_kwargs(params, metavar="PARAMS")

    from xcube.util.dsio import guess_dataset_format
    format_name = format if format else guess_dataset_format(output)

    from xcube.api import open_dataset, chunk_dataset, write_dataset

    with open_dataset(input_path=cube) as ds:
        if chunk_sizes:
            for k in chunk_sizes:
                if k not in ds.dims:
                    raise click.ClickException("Invalid value for CHUNKS, "
                                               f"{k!r} is not the name of any dimension: {chunks}")

        chunked_dataset = chunk_dataset(ds, chunk_sizes=chunk_sizes, format_name=format_name)
        write_dataset(chunked_dataset, output_path=output, format_name=format_name, **write_kwargs)
