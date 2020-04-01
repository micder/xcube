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

import logging
import time
import threading

import click
import flask
import numpy as np

DEFAULT_TILE_SIZE = 1024
DEFAULT_PORT = 3030


def run_tile_server(tile_size: int = DEFAULT_TILE_SIZE,
                    host: str = None,
                    port: int = DEFAULT_PORT,
                    debug: bool = False):
    """
    Start the test tile server.

    :param tile_size: Tile size to be used.
    :param host: The hostname to listen on. Set this to ``'0.0.0.0'`` to
        have the server available externally as well. Defaults to ``'127.0.0.1'``.
    :param port: The port to listen on. Defaults to ``5000``.
    :param debug: If given, enable or disable debug mode.
    """
    tile_size = tile_size or DEFAULT_TILE_SIZE
    port = port or DEFAULT_PORT

    app = flask.Flask(__name__)

    app.logger.setLevel(logging.INFO)

    start_time = time.perf_counter()

    def get_millis_since_reset() -> int:
        nonlocal start_time
        return round(1000 * (time.perf_counter() - start_time))

    @app.route('/tile/<int:z>/<int:y>/<int:x>.data', methods=['GET'])
    def _tile(z: int, y: int, x: int):
        nonlocal start_time
        t1 = get_millis_since_reset()
        r = np.random.RandomState(seed=(10 * (z + 10 * y) + x))
        for i in range(10):
            data = r.standard_normal(tile_size * tile_size)
        data = data.tobytes()
        t2 = get_millis_since_reset()
        app.logger.info(f'tile {z}/{y}/{x}, {t1}, {t2}, {t2 - t1}, {threading.current_thread()}')
        return data

    app.run(host=host, port=port, debug=debug, threaded=True)


@click.command(name="tile_server")
@click.option('--tile-size', help=f'Tile size. Defaults to {DEFAULT_TILE_SIZE}.',
              type=int, default=DEFAULT_TILE_SIZE)
@click.option('--host', '-a',
              help="The host address to listen on. "
                   "Set this to '0.0.0.0' to have the server available externally as well. "
                   "Defaults to  '127.0.0.1'.")
@click.option('--port', '-p', type=int, default=DEFAULT_PORT,
              help=f"The port number to listen on. Defaults to {DEFAULT_PORT}.")
@click.option('--debug', is_flag=True, help='Output extra debugging information.')
def tile_server(tile_size: int = None,
                host: str = None,
                port: int = None,
                debug: bool = False):
    """
    Start the test tile server.
    """
    run_tile_server(tile_size=tile_size, host=host, port=port, debug=debug)


if __name__ == '__main__':
    tile_server.main()
