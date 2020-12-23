import json
import unittest
from collections import MutableMapping
from typing import Dict, KeysView, Iterator

import numcodecs as nc
import numpy as np
import xarray as xr

ZARR_PATH = 'where_test.zarr'


class WhereTest(unittest.TestCase):

    # def tearDown(self) -> None:
    #     if os.path.exists(ZARR_PATH):
    #         shutil.rmtree(ZARR_PATH)

    def test_xarray_where(self):
        shape = (8, 8)
        chunks = (4, 4)
        array_dtype = np.dtype(np.int32)
        compressor_kwargs = {
            "blocksize": 0,
            "clevel": 5,
            "cname": "lz4",
            "shuffle": 1
        }
        array_zarray = {
            "zarr_format": 2,
            "shape": shape,
            "chunks": chunks,
            "dtype": str(array_dtype.str),
            "order": "C",
            "compressor": {
                "id": "blosc",
                **compressor_kwargs,
            },
            "fill_value": None,
            "filters": None,
        }
        array_zattrs = {
            "_ARRAY_DIMENSIONS": [
                "y",
                "x"
            ]
        }
        mask_0 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=array_dtype)
        mask_1 = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=array_dtype)
        mask_x = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ], dtype=array_dtype)
        store = {
            '.zgroup': {
                'zarr_format': 2
            },
            '.zattrs': {},
            'mask/.zarray': array_zarray,
            'mask/.zattrs': array_zattrs,
            'mask/0.0': mask_0,
            'mask/0.1': mask_x,
            'mask/1.0': mask_x,
            'mask/1.1': mask_1,
            'data/.zarray': {**array_zarray, 'fill_value': 0},
            'data/.zattrs': array_zattrs,
            'data/0.0': np.random.randint(10, 99, size=chunks, dtype=array_dtype),
            'data/0.1': np.random.randint(10, 99, size=chunks, dtype=array_dtype),
            # 'data/1.0': missing!
            'data/1.1': np.random.randint(10, 99, size=chunks, dtype=array_dtype),
        }

        compressor = nc.blosc.Blosc(**compressor_kwargs)
        encoded_store = dict()
        for k, v in store.items():
            if isinstance(v, np.ndarray):
                encoded_store[k] = compressor.encode(v.tobytes(order='C'))
            else:
                encoded_store[k] = json.dumps(v).encode('utf-8')

        store = DictStore(encoded_store)
        ds = xr.open_zarr(store)
        requested_keys_before_xwhere = store.requested_keys()
        masked_data = xwhere(ds.mask, ds.data)
        masked_data_values = masked_data.values
        requested_keys_after_xwhere = store.requested_keys()

        print('data:\n', ds.data.values)
        print('mask:\n', ds.mask.values)
        print('masked data:\n', masked_data_values)

        self.assertEqual(
            {
                '.zgroup', '.zattrs',
                'data/.zarray', 'data/.zattrs',
                'mask/.zarray', 'mask/.zattrs',
            },
            requested_keys_before_xwhere)

        self.assertEqual(
            {
                '.zgroup', '.zattrs',
                'data/.zarray', 'data/.zattrs',
                # 'data/0.0', Should not be in, because mask/0.0 is all zero
                'data/0.1',
                'data/1.0',
                'data/1.1',
                'mask/.zarray', 'mask/.zattrs',
                'mask/0.0',
                'mask/0.1',
                'mask/1.0',
                'mask/1.1'
            },
            requested_keys_after_xwhere)


def xwhere(cond: xr.DataArray, x: xr.DataArray, y: xr.DataArray = None):
    return xr.where(cond, x, y if y is not None else float('nan'))


class DictStore(MutableMapping):
    """
    A MutableMapping that is NOT a dict.
    All interface operations are delegated to a wrapped dict instance.
    :param entries: optional entries
    """

    def __init__(self, entries: Dict[str, bytes] = None):
        self._entries: Dict[str, bytes] = dict(entries) if entries else dict()
        self._requested_keys = set()

    def requested_keys(self):
        return set(self._requested_keys)

    def keys(self) -> KeysView[str]:
        return self._entries.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> bytes:
        self._requested_keys.add(key)
        return self._entries[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self._entries[key] = value

    def __delitem__(self, key: str) -> None:
        del self._entries[key]
