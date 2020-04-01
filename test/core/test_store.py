import threading
import unittest
import warnings
from typing import Tuple

import numpy as np
import xarray as xr

from xcube.core.store import CubeStore


class CubeStoreTest(unittest.TestCase):

    def test_concurrency_is_provided(self):

        threads = set()
        indexes = list()
        errors = list()

        def get_chunk(cube_store: CubeStore, name: str, index: Tuple[int, ...]) -> bytes:
            # print('get_chunk: ', name, index, threading.current_thread())
            try:
                print()
                threads.add(threading.current_thread())
                indexes.append(index)
                data = np.zeros(cube_store.chunks, dtype=np.float64)
                data_view = data.ravel()
                return data_view.tobytes()
            except BaseException as error:
                errors.append(error)

        store = CubeStore(dims=('time', 'y', 'x'), shape=(4, 8, 16), chunks=(2, 4, 8))
        store.add_lazy_array('TEST', '<f8', get_chunk=get_chunk)

        print(f'len(threads) = {len(threads)}')
        ds = xr.open_zarr(store)
        print(f'len(threads) = {len(threads)}')
        try:
            var = ds.TEST.compute()
        except BaseException as e:
            print(e)
        print(f'len(threads) = {len(threads)}')

        self.assertEqual((4, 8, 16), var.shape)
        self.assertEqual(2 * 2 * 2, len(indexes))
        self.assertEqual(0, len(errors))
        print(f'len(threads) = {len(threads)}')
        self.assertTrue(2 <= len(threads) <= 8, msg=f'was {len(threads)}')


    def test_lazy_index_var(self):
        index_var = gen_index_var(dims=('time', 'lat', 'lon'),
                                  shape=(4, 8, 16),
                                  chunks=(2, 4, 8))
        self.assertIsNotNone(index_var)
        self.assertEqual((4, 8, 16), index_var.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), index_var.chunks)
        self.assertEqual(('time', 'lat', 'lon'), index_var.dims)

        visited_indexes = set()

        def index_var_ufunc(index_var_):
            if index_var_.size < 6:
                warnings.warn(f"weird variable of size {index_var_.size} received!")
                return
            nonlocal visited_indexes
            index = tuple(map(int, index_var_.ravel()[0:6]))
            visited_indexes.add(index)
            return index_var_

        result = xr.apply_ufunc(index_var_ufunc, index_var, dask='parallelized', output_dtypes=[index_var.dtype])
        self.assertIsNotNone(result)
        self.assertEqual((4, 8, 16), result.shape)
        self.assertEqual(((2, 2), (4, 4), (8, 8)), result.chunks)
        self.assertEqual(('time', 'lat', 'lon'), result.dims)

        values = result.values
        self.assertEqual((4, 8, 16), values.shape)
        np.testing.assert_array_equal(index_var.values, values)

        self.assertEqual(8, len(visited_indexes))
        self.assertEqual({
            (0, 2, 0, 4, 0, 8),
            (2, 4, 0, 4, 0, 8),
            (0, 2, 4, 8, 0, 8),
            (2, 4, 4, 8, 0, 8),
            (0, 2, 0, 4, 8, 16),
            (2, 4, 0, 4, 8, 16),
            (0, 2, 4, 8, 8, 16),
            (2, 4, 4, 8, 8, 16),
        }, visited_indexes)


def gen_index_var(dims, shape, chunks):
    # noinspection PyUnusedLocal
    def get_chunk(cube_store: CubeStore, name: str, index: Tuple[int, ...]) -> bytes:
        data = np.zeros(cube_store.chunks, dtype=np.uint64)
        data_view = data.ravel()
        if data_view.base is not data:
            raise ValueError('view expected')
        if data_view.size < cube_store.ndim * 2:
            raise ValueError('size too small')
        for i in range(cube_store.ndim):
            j1 = cube_store.chunks[i] * index[i]
            j2 = j1 + cube_store.chunks[i]
            data_view[2 * i] = j1
            data_view[2 * i + 1] = j2
        return data.tobytes()

    store = CubeStore(dims, shape, chunks)
    store.add_lazy_array('__index_var__', '<u8', get_chunk=get_chunk)

    ds = xr.open_zarr(store)
    return ds.__index_var__
