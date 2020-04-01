import asyncio
import itertools
import json
import os
import os.path
import subprocess
import sys
import time
from typing import Tuple

import aiohttp
import click
import numcodecs
import numpy as np
import requests
import xarray as xr

from xcube.core.dsio import rimraf
from xcube.core.store import CubeStore

DEFAULT_TILE_SIZE = 1024

NUM_X_TILES = 5
NUM_Y_TILES = 5
NUM_TIMES = 10


def test_async(port: int, tile_size: int, dataset_path: str, var_name: str):
    """
    Code inspired by https://pawelmhm.github.io/asyncio/python/aiohttp/2016/04/22/asyncio-aiohttp.html
    """

    var_path = os.path.join(dataset_path, var_name)
    os.makedirs(var_path, exist_ok=True)

    with open(os.path.join(dataset_path, '.zgroup'), 'w') as fp:
        json.dump(
            {
                "zarr_format": 2
            },
            fp, indent=2)
    with open(os.path.join(dataset_path, '.zattrs'), 'w') as fp:
        json.dump({}, fp, indent=2)
    with open(os.path.join(var_path, '.zarray'), 'w') as fp:
        json.dump(
            {
                "chunks": [
                    1,
                    tile_size,
                    tile_size
                ],
                "compressor": None,
                "dtype": "<f8",
                "fill_value": "NaN",
                "filters": None,
                "order": "C",
                "shape": [
                    NUM_TIMES,
                    tile_size * NUM_Y_TILES,
                    tile_size * NUM_X_TILES
                ],
                "zarr_format": 2
            },
            fp, indent=2)
    with open(os.path.join(var_path, '.zattrs'), 'w') as fp:
        json.dump(
            {
                "_ARRAY_DIMENSIONS": [
                    "time",
                    "y",
                    "x"
                ]
            },
            fp, indent=2)

    async def fetch(url, session):
        async with session.get(url) as response:
            return await response.read()

    async def bound_fetch(session, sem, url, chunk_name):
        # Getter function with semaphore.
        async with sem:
            data = await fetch(url, session)
            write_data(data, chunk_name)

    def write_data(data, chunk_name):
        # TODO: configure .zarray, so we can do:
        # blosc = numcodecs.Blosc()
        # data = blosc.encode(data)
        chunk_path = os.path.join(var_path, chunk_name)
        with open(chunk_path, 'wb') as fp:
            fp.write(data)

    async def run():
        requests = [(_make_tile_url(port, z, y, x), f'{z}.{y}.{x}')
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
            for url, chunk_name in requests:
                # pass Semaphore and session to every GET request
                task = asyncio.ensure_future(bound_fetch(session, sem, url, chunk_name))
                tasks.append(task)
            responses = asyncio.gather(*tasks)
            await responses

    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(run())
    print(f'Fetching & writing...')
    loop.run_until_complete(future)


def test_store(port: int, tile_size: int, dataset_path: str, var_name: str):
    zero_tile = np.zeros(tile_size * tile_size, dtype=np.float64).tobytes()

    def get_chunk(cube_store: CubeStore, name: str, index: Tuple[int, ...]) -> bytes:
        z, y, x = index
        response = requests.get(_make_tile_url(port, z, y, x))
        if response.ok:
            return response.content
        else:
            return zero_tile

    store = CubeStore(dims=('time', 'y', 'x'),
                      shape=(NUM_TIMES, NUM_Y_TILES * tile_size, NUM_X_TILES * tile_size),
                      chunks=(1, tile_size, tile_size))
    store.add_lazy_array(var_name, '<f8', get_chunk=get_chunk)
    ds = xr.open_zarr(store)
    print(f'Writing {dataset_path}...')
    ds.to_zarr(dataset_path)


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

        dataset_path = f'tile_client_{mode}_{tile_size}.zarr'
        if os.path.exists(dataset_path):
            print(f'Removing {dataset_path}...')
            rimraf(dataset_path)

        var_name = 'TEST'

        t1 = time.perf_counter()
        if mode == 'async':
            test_async(port, tile_size, dataset_path, var_name)
        else:
            test_store(port, tile_size, dataset_path, var_name)
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
