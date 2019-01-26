import os
import unittest

import numpy as np
import xarray as xr

from test.sampledata import new_test_cube, new_test_dataset
from xcube.api import open_dataset, read_dataset, write_dataset, dump_dataset, chunk_dataset, validate_cube, \
    assert_cube, get_cube_point_indexes, get_cube_point_values

TEST_NC_FILE = "test.nc"


class OpenReadWriteDatasetTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        dataset = new_test_cube()
        dataset.to_netcdf(TEST_NC_FILE, mode="w")
        dataset.close()

    def tearDown(self):
        os.remove(TEST_NC_FILE)
        super().tearDown()

    def test_open_dataset(self):
        with open_dataset(TEST_NC_FILE) as ds:
            self.assertIsNotNone(ds)

    def test_read_dataset(self):
        ds = read_dataset(TEST_NC_FILE)
        self.assertIsNotNone(ds)
        ds.close()

    def test_write_dataset(self):
        TEST_NC_FILE_2 = "test-2.nc"

        dataset = new_test_cube()
        try:
            write_dataset(dataset, TEST_NC_FILE_2)
            self.assertTrue(os.path.isfile(TEST_NC_FILE_2))
        finally:
            if os.path.isfile(TEST_NC_FILE_2):
                os.remove(TEST_NC_FILE_2)


class ChunkDatasetTest(unittest.TestCase):
    def test_chunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=10, lon=20),
                                        format_name="zarr")
        self.assertEqual({'chunks': (1, 10, 20)}, chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunks': (1, 10, 20)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=20, lon=40),
                                        format_name="netcdf4")
        self.assertEqual({'chunksizes': (1, 20, 40)}, chunked_dataset.precipitation.encoding)
        self.assertEqual({'chunksizes': (1, 20, 40)}, chunked_dataset.temperature.encoding)

        chunked_dataset = chunk_dataset(dataset,
                                        chunk_sizes=dict(time=1, lat=20, lon=40))
        self.assertEqual({}, chunked_dataset.precipitation.encoding)
        self.assertEqual({}, chunked_dataset.temperature.encoding)

    def test_unchunk_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        for var in dataset.data_vars.values():
            var.encoding.update({"chunks": (5, 180, 360), "_FillValue": -999.0})

        chunked_dataset = chunk_dataset(dataset, format_name="zarr")
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.precipitation.encoding)
        self.assertEqual({"_FillValue": -999.0}, chunked_dataset.temperature.encoding)


class DumpDatasetTest(unittest.TestCase):
    def test_dump_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)
        for var in dataset.variables.values():
            var.encoding.update({"_FillValue": 999.0})

        print(dataset.dims)

        text = dump_dataset(dataset)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (lat: 180, lon: 360, time: 5)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertNotIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertNotIn("Encoding for data variable 'temperature':\n", text)
        self.assertNotIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, show_var_encoding=True)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (lat: 180, lon: 360, time: 5)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertIn("Encoding for data variable 'temperature':\n", text)
        self.assertIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, ["precipitation"])
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertNotIn("Encoding:\n", text)
        self.assertNotIn("    _FillValue:  999.0", text)

        text = dump_dataset(dataset, ["precipitation"], show_var_encoding=True)
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertIn("Encoding:\n", text)
        self.assertIn("    _FillValue:  999.0", text)


class AssertAndValidateCubeTest(unittest.TestCase):

    def test_assert_cube(self):
        cube = new_test_cube()
        cube["chl"] = xr.DataArray(np.random.rand(cube.dims["lat"], cube.dims["lon"]),
                                   dims=("lat", "lon"),
                                   coords=dict(lat=cube.lat, lon=cube.lon))
        with self.assertRaises(ValueError) as cm:
            assert_cube(cube)
        self.assertEqual("Dataset is not a valid data cube, because:\n"
                         "- dimensions of data variable 'chl' must be"
                         " ('time', ..., 'lat', 'lon'), but were ('lat', 'lon') for 'chl';\n"
                         "- dimensions of all data variables must be same,"
                         " but found ('time', 'lat', 'lon') for 'precipitation'"
                         " and ('lat', 'lon') for 'chl'.",
                         f"{cm.exception}")

    def test_validate_cube(self):
        cube = new_test_cube()
        self.assertEqual([], validate_cube(cube))


class ExtractPointsTest(unittest.TestCase):
    def test_get_cube_point_values(self):
        cube = new_test_cube()
        values = get_cube_point_values(cube,
                                       dict(time=np.array(["2010-01-04", "2010-01-02",
                                                           "2010-01-08", "2010-01-02",
                                                           "2010-01-02", "2010-01-01",
                                                           "2010-01-05", "2010-01-03",
                                                           ], dtype="datetime64[ns]"),
                                            lat=np.array([50.0, 51.3, 49.7, 50.1, 51.9, 50.8, 50.2, 52.0]),
                                            lon=np.array([0.0, 0.1, 0.4, 2.9, 1.6, 0.7, -0.5, 4.0]),
                                            ))
        print(values)

    def test_get_cube_point_indexes(self):
        cube = new_test_cube()
        indexes = get_cube_point_indexes(cube,
                                         dict(time=np.array(["2010-01-04", "2010-01-02",
                                                             "2010-01-08", "2010-01-02",
                                                             "2010-01-02", "2010-01-01",
                                                             "2010-01-05", "2010-01-03",
                                                             ], dtype="datetime64[ns]"),
                                              lat=np.array([50.0, 51.3, 49.7, 50.1, 51.9, 50.8, 50.2, 52.0]),
                                              lon=np.array([0.0, 0.1, 0.4, 2.9, 1.6, 0.7, -0.5, 4.0]),
                                              ))

        self.assertEqual(["time", "lat", "lon"], [c for c in indexes])
        np.testing.assert_array_equal(
            np.array([[3, 0, 0],
                      [1, 64, 4],
                      [-1, -1, 19],
                      [1, 4, 144],
                      [1, 94, 79],
                      [0, 39, 34],
                      [4, 9, -1],
                      [2, 99, 199]]),
            np.stack([indexes[c] for c in indexes], axis=-1))
