import asyncio
import itertools
import os.path
import subprocess
import sys
import time
from typing import Tuple

import aiohttp
import click
import numpy as np
import requests
import xarray as xr

from xcube.core.store import CubeStore

DEFAULT_TILE_SIZE = 1024

NUM_X_TILES = 5
NUM_Y_TILES = 5
NUM_TIMES = 10


def test_async_fetch(tile_size: int, port: int):
    """
    Code inspired by https://pawelmhm.github.io/asyncio/python/aiohttp/2016/04/22/asyncio-aiohttp.html
    """

    async def fetch(url, session):
        async with session.get(url) as response:
            return await response.read()

    async def bound_fetch(sem, url, session):
        # Getter function with semaphore.
        async with sem:
            await fetch(url, session)

    async def run():
        requests = [_make_tile_url(port, z, y, x)
                    for z, y, x in itertools.product(range(0, NUM_TIMES),
                                                     range(0, NUM_Y_TILES),
                                                     range(0, NUM_X_TILES))]
        tasks = []

        # operating systems limit the total number of open sockets (files) allowed
        # therefore we create an instance of Semaphore allowing only 1000 sockets
        sem = asyncio.Semaphore(1000)

        # Create client session that will ensure we don't open new connection
        # per each request.
        async with aiohttp.ClientSession() as session:
            for request in requests:
                # pass Semaphore and session to every GET request
                task = asyncio.ensure_future(bound_fetch(sem, request, session))
                tasks.append(task)
            responses = asyncio.gather(*tasks)
            await responses

    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(run())
    print(f'Fetching...')
    loop.run_until_complete(future)


def test_store(tile_size: int, port: int):
    zero_tile = np.zeros(tile_size * tile_size, dtype=np.float64).tobytes()

    def get_chunk(cube_store: CubeStore, name: str, index: Tuple[int, ...]) -> bytes:
        z, y, x = index
        response = requests.get(_make_tile_url(port, z, y, x))
        if response.ok:
            return response.content
        else:
            return zero_tile

    store = CubeStore(dims=('time', 'y', 'x'),
                      shape=(5, 4 * tile_size, 4 * tile_size),
                      chunks=(1, tile_size, tile_size))
    store.add_lazy_array('TEST', '<f8', get_chunk=get_chunk)
    ds = xr.open_zarr(store)
    print(f'Writing Zarr...')
    ds.to_zarr('tile_client.zarr')


def _make_tile_url(port, z, y, x):
    return f'http://localhost:{port}/tile/{z}/{y}/{x}.data'


@click.command(name="tile_client")
@click.argument('mode', type=click.Choice(['async', 'store']))
@click.option('--tile-size', help=f'Tile size. Defaults to {DEFAULT_TILE_SIZE}.',
              type=int, default=DEFAULT_TILE_SIZE)
def tile_client(mode: str,
                tile_size: int = None):
    """
    Run the test tile client.
    """
    port = 9092
    with subprocess.Popen([sys.executable,
                           os.path.join(os.path.dirname(__file__), 'tile_server.py'),
                           '--tile-size',
                           str(tile_size),
                           '--port',
                           str(port)]) as process:
        t1 = time.perf_counter()
        if mode == 'async':
            test_async_fetch(tile_size, port)
        else:
            test_store(tile_size, port)
        t2 = time.perf_counter()

        process.kill()

        num_bytes_per_value = 8
        time_total = t2 - t1
        num_bytes_total = NUM_TIMES * (NUM_Y_TILES * tile_size) * (NUM_X_TILES * tile_size) * num_bytes_per_value
        bytes_per_sec = num_bytes_total / time_total

        print()
        print(f'num_bytes_per_value = {num_bytes_per_value}')
        print(f'num_times = {NUM_TIMES}')
        print(f'num_y_tiles = {NUM_Y_TILES}')
        print(f'num_x_tiles = {NUM_X_TILES}')
        print(f'tile_size = {tile_size}')
        print()
        print(f'Done after {time_total} sec.')
        print(f'Bytes read: {num_bytes_total} ({num_bytes_total // (1024 * 1024)} MiB)')
        print(f'Rate: {bytes_per_sec} ({bytes_per_sec // (1024 * 1024)} MiB/sec)')


if __name__ == '__main__':
    tile_client.main()
