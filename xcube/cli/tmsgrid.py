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

import fractions
import math
from typing import List, Optional, Tuple

import click

from xcube.constants import EARTH_EQUATORIAL_PERIMETER

_DEFAULT_TILE_SIZE = 256

_EARTH_EQUATORIAL_PERIMETER_FRACTION = fractions.Fraction(EARTH_EQUATORIAL_PERIMETER)

_UNITS = {
    'deg': fractions.Fraction(1),
    'cm': fractions.Fraction(360 // 100, _EARTH_EQUATORIAL_PERIMETER_FRACTION),
    'km': fractions.Fraction(360 * 1000, _EARTH_EQUATORIAL_PERIMETER_FRACTION),
    'm': fractions.Fraction(360, _EARTH_EQUATORIAL_PERIMETER_FRACTION),
}

_RESOLUTION_FORMAT = f'Num[/Denom][{"|".join(_UNITS.keys())}]'


def _parse_res(res_str: str) -> Tuple[fractions.Fraction, str]:
    unit = 'deg'
    factor = fractions.Fraction(1)
    for u, f in _UNITS.items():
        if res_str.endswith(u):
            res_str = res_str[0:-len(u)]
            unit = u
            factor = f
            break
    try:
        res = fractions.Fraction(res_str)
    except ValueError as e:
        raise click.ClickException(f'RESOLUTION must use format {_RESOLUTION_FORMAT}.') from e
    if res <= 0:
        raise click.ClickException('RESOLUTION must be greater than zero.')
    return res * factor, unit


def get_resolutions(max_level: int, tile_size: int = _DEFAULT_TILE_SIZE) -> List[fractions.Fraction]:
    return [get_resolution_for_level(level, tile_size=tile_size) for level in range(0, max_level + 1)]


def get_resolution_for_level(level: int, tile_size: int = _DEFAULT_TILE_SIZE) -> fractions.Fraction:
    return fractions.Fraction(180, tile_size << level)


def get_level_for_resolution(res: fractions.Fraction, tile_size: int = _DEFAULT_TILE_SIZE) -> int:
    return round(math.log2(180 / tile_size / res))


def get_tile_grid_for_level(level: int) -> Tuple[int, int]:
    return 2 << level, 1 << level


def _parse_res_or_level(res: Optional[str], level: Optional[int], tile_size: int) -> Tuple[
    fractions.Fraction, str, int]:
    if res is None and level is None:
        raise click.ClickException('One of RESOLUTION or LEVEL must be given.')
    elif res is not None and level is not None:
        raise click.ClickException('Only one of RESOLUTION and LEVEL can be given.')
    elif res is not None:
        res_deg, unit = _parse_res(res)
        level = get_level_for_resolution(res_deg, tile_size=tile_size)
        return res_deg, unit, level
    else:
        res_deg, unit = get_resolution_for_level(level, tile_size), 'deg'
        return res_deg, unit, level


RESOLUTION_OPTION = click.option('--res', '-r', metavar="RESOLUTION",
                                 help=f'The grid resolution. General form: {_RESOLUTION_FORMAT}.')
LEVEL_OPTION = click.option('--level', '-l', metavar="LEVEL",
                            help='The grid level.', type=int)
TILE_SIZE_OPTION = click.option('--tile-size', '-t', default=_DEFAULT_TILE_SIZE,
                                help=f'Tile size. Defaults to {_DEFAULT_TILE_SIZE}.')


@click.command(name="bbox")
@click.argument('geom', metavar="GEOMETRY")
@RESOLUTION_OPTION
@LEVEL_OPTION
@TILE_SIZE_OPTION
def bbox(geom: str, res: Optional[str], level: Optional[int], tile_size: int):
    res_deg, unit, level = _parse_res_or_level(res, level, tile_size)
    # TODO


@click.command(name="ls")
@RESOLUTION_OPTION
@LEVEL_OPTION
@TILE_SIZE_OPTION
@click.option('--all', '-a', is_flag=True, help='Output also lower levels.')
def ls(res: Optional[str], level: Optional[int], tile_size: int, all: bool):
    res_deg, unit, level = _parse_res_or_level(res, level, tile_size)
    resolutions = get_resolutions(level, tile_size=tile_size)
    for z in range(len(resolutions)):
        res_deg = resolutions[z]
        res_m = res_deg / _UNITS['m']
        if all or z == level:
            print(f'Level = {z}, Resolution = {float(res_deg)} deg = {float(res_m)} m')


@click.group()
def tmsgrid():
    pass


tmsgrid.add_command(ls)
tmsgrid.add_command(bbox)
