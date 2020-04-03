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
import threading
import time

import click
import flask
import numpy as np
import tornado.ioloop
import tornado.web

from xcube.webapi.service import url_pattern

DEFAULT_SERVER = 'tornado'
DEFAULT_TILE_SIZE = 1024
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 9092
DEFAULT_NAME = 'test'


class Context:
    def __init__(self,
                 name: str = DEFAULT_NAME,
                 tile_size: int = DEFAULT_TILE_SIZE,
                 host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT,
                 debug: bool = False):
        self.name = name
        self.tile_size = tile_size
        self.host = host
        self.port = port
        self.debug = debug
        self.start_time = time.perf_counter()
        self.record_index = 0
        self.records_fp = open(f'tile_server_{name}_{tile_size}.csv', 'w')
        self.records_fp.write('idx,tile,t1,t2,dt')
        self.records_fp.flush()

    def get_tile(self, z: int, y: int, x: int):
        t1 = self._get_millis_since_start()
        data = self._compute_tile(z, y, x)
        t2 = self._get_millis_since_start()

        self.record_index += 1
        record = self.record_index, f'{z}/{y}/{x}', t1, t2, t2 - t1, threading.current_thread().getName()
        record_line = ','.join(map(str, record))
        self.records_fp.write('\n' + record_line)
        self.records_fp.flush()

        print(record, flush=True)

        return data

    def _compute_tile(self, z: int, y: int, x: int) -> bytes:
        tile_size = self.tile_size
        r = np.random.RandomState(seed=(10 * (z + 10 * y) + x))
        data = r.standard_normal(tile_size * tile_size)
        for i in range(9):
            data = r.standard_normal(tile_size * tile_size)
        return data.tobytes()

    def _get_millis_since_start(self) -> int:
        return round(1000 * (time.perf_counter() - self.start_time))


def run_tile_server_tornado(context: Context):
    class TileHandler(tornado.web.RequestHandler):
        async def get(self, z: str, y: str, x: str):
            data = await tornado.ioloop.IOLoop.current().run_in_executor(None,
                                                                         context.get_tile,
                                                                         int(z), int(y), int(x))
            self.set_header('Content-Type', 'application/octet-stream')
            await self.finish(data)

    app = tornado.web.Application([
        (url_pattern('/tile/{{z}}/{{y}}/{{x}}.data'), TileHandler)
    ])
    app.listen(context.port, address=context.host)
    tornado.ioloop.IOLoop.current().start()


def run_tile_server_flask(context: Context):
    app = flask.Flask(__name__)
    app.logger.setLevel(logging.INFO)

    @app.route('/tile/<int:z>/<int:y>/<int:x>.data', methods=['GET'])
    def _tile(z: int, y: int, x: int):
        return context.get_tile(z, y, x)

    app.run(host=context.host, port=context.port, debug=context.debug, threaded=True)


@click.command(name="tile_server")
@click.option('--server', help=f'Server type. Defaults to {DEFAULT_SERVER!r}.',
              type=click.Choice(['tornado', 'flask']))
@click.option('--name', help=f'Test name. Defaults to {DEFAULT_NAME!r}.',
              default=DEFAULT_NAME)
@click.option('--tile-size', help=f'Tile size. Defaults to {DEFAULT_TILE_SIZE}.',
              type=int, default=DEFAULT_TILE_SIZE)
@click.option('--host',
              help="The host address to listen on. "
                   "Set this to '0.0.0.0' to have the server available externally as well. "
                   f"Defaults to  {DEFAULT_HOST!r}.", default=DEFAULT_HOST)
@click.option('--port', type=int, default=DEFAULT_PORT,
              help=f"The port number to listen on. Defaults to {DEFAULT_PORT}.")
@click.option('--debug', is_flag=True, help='Output extra debugging information.')
def tile_server(server: str = None,
                name: str = None,
                tile_size: int = None,
                host: str = None,
                port: int = None,
                debug: bool = False):
    """
    Start the test tile server.
    """
    context = Context(name=name, tile_size=tile_size, host=host, port=port, debug=debug)
    if server == 'flask':
        run_tile_server_flask(context)
    else:
        run_tile_server_tornado(context)


if __name__ == '__main__':
    tile_server.main()
