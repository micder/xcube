import unittest

import xarray as xr

from xcube.core.new import new_cube
from xcube.core.store import DataStoreError
from xcube.core.store import DatasetDescriptor
from xcube.core.store import TYPE_SPECIFIER_CUBE
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import new_data_store
from xcube.core.store.stores.memory import MemoryDataStore
from xcube.util.jsonschema import JsonObjectSchema


class MemoryCubeStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.old_global_data_dict = MemoryDataStore.replace_global_data_dict({
            'cube_1': new_cube(variables=dict(B01=0.4, B02=0.5)),
            'cube_2': new_cube(variables=dict(B03=0.4, B04=0.5)),
            'ds_1': xr.Dataset()
        })
        self._store = new_data_store('memory')
        self.assertIsInstance(self.store, MemoryDataStore)

    @property
    def store(self) -> MemoryDataStore:
        # noinspection PyTypeChecker
        return self._store

    def tearDown(self) -> None:
        MemoryDataStore.replace_global_data_dict(self.old_global_data_dict)

    def test_get_type_specifiers(self):
        self.assertEqual(('*',), self.store.get_type_specifiers())

    def test_get_type_specifiers_for_data(self):
        self.assertEqual(('dataset[cube]', ), self.store.get_type_specifiers_for_data('cube_1'))
        self.assertEqual(('dataset', ), self.store.get_type_specifiers_for_data('ds_1'))
        with self.assertRaises(DataStoreError) as cm:
            self.store.get_type_specifiers_for_data('geodataframe_2')
        self.assertEqual('Data resource "geodataframe_2" does not exist in store', f'{cm.exception}')

    def test_get_data_ids(self):
        self.assertEqual({('cube_1', None), ('cube_2', None), ('ds_1', None)}, set(self.store.get_data_ids()))
        self.assertEqual({('cube_1', None), ('cube_2', None), ('ds_1', None)},
                         set(self.store.get_data_ids(include_titles=False)))

    def test_has_data(self):
        self.assertEqual(True, self.store.has_data('cube_1'))
        self.assertEqual(False, self.store.has_data('cube_3'))
        self.assertEqual(True, self.store.has_data('cube_1', type_specifier='dataset'))
        self.assertEqual(True, self.store.has_data('cube_1', type_specifier='dataset[cube]'))
        self.assertEqual(False, self.store.has_data('cube_1', type_specifier='dataset[multilevel]'))
        self.assertEqual(True, self.store.has_data('ds_1', type_specifier='dataset'))
        self.assertEqual(False, self.store.has_data('ds_1', type_specifier='dataset[cube]'))

    def test_describe_data(self):
        dd = self.store.describe_data('cube_1')
        self.assertIsInstance(dd, DatasetDescriptor)
        self.assertEqual(
            DatasetDescriptor(
                data_id='cube_1',
                type_specifier=TYPE_SPECIFIER_CUBE,
            ).to_dict(),
            dd.to_dict())

        dd = self.store.describe_data('cube_1', type_specifier='dataset[cube]')
        self.assertIsInstance(dd, DatasetDescriptor)
        self.assertEqual(
            DatasetDescriptor(
                data_id='cube_1',
                type_specifier=TYPE_SPECIFIER_CUBE,
            ).to_dict(),
            dd.to_dict())

    def test_get_search_params_schema(self):
        schema = self.store.get_search_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertEqual({}, schema.properties)
        self.assertEqual(False, schema.additional_properties)

        schema = self.store.get_search_params_schema(type_specifier='geodataframe')
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertEqual({}, schema.properties)
        self.assertEqual(False, schema.additional_properties)

    def test_search_data(self):
        result = list(self.store.search_data(type_specifier=TYPE_SPECIFIER_DATASET))
        self.assertEqual(3, len(result))
        self.assertIsInstance(result[0], DatasetDescriptor)
        self.assertEqual(result[0].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertIsInstance(result[1], DatasetDescriptor)
        self.assertEqual(result[1].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertIsInstance(result[2], DatasetDescriptor)
        self.assertEqual(result[2].type_specifier, TYPE_SPECIFIER_DATASET)

        result = list(self.store.search_data(type_specifier=TYPE_SPECIFIER_CUBE))
        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], DatasetDescriptor)
        self.assertEqual(result[0].type_specifier, TYPE_SPECIFIER_CUBE)
        self.assertIsInstance(result[1], DatasetDescriptor)
        self.assertEqual(result[1].type_specifier, TYPE_SPECIFIER_CUBE)

        with self.assertRaises(DataStoreError) as cm:
            list(self.store.search_data(type_specifier=TYPE_SPECIFIER_DATASET,
                                        time_range=['2020-03-01', '2020-03-04'], bbox=[52, 11, 54, 12]))
        self.assertEqual('Unsupported search parameters: time_range, bbox', f'{cm.exception}')

    def test_get_data_opener_ids(self):
        self.assertEqual(('*:*:memory',), self.store.get_data_opener_ids())
        self.assertEqual(('*:*:memory',), self.store.get_data_opener_ids('dataset'))
        self.assertEqual(('*:*:memory',), self.store.get_data_opener_ids('dataset[cube]'))
        self.assertEqual(('*:*:memory',), self.store.get_data_opener_ids('geodataframe'))

    def test_get_open_data_params_schema(self):
        schema = self.store.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertEqual({}, schema.properties)

    def test_open_data(self):
        cube_1 = self.store.open_data('cube_1')
        self.assertIsInstance(cube_1, xr.Dataset)
        self.assertEqual({'B01', 'B02'}, set(map(str, cube_1.data_vars.keys())))
        cube_2 = self.store.open_data('cube_2')
        self.assertIsInstance(cube_2, xr.Dataset)
        self.assertEqual({'B03', 'B04'}, set(map(str, cube_2.data_vars.keys())))
        with self.assertRaises(DataStoreError) as cm:
            self.store.open_data('cube_3')
        self.assertEqual('Data resource "cube_3" does not exist in store', f'{cm.exception}')
        with self.assertRaises(DataStoreError) as cm:
            self.store.open_data('cube_1', tile_size=1000, spatial_res=0.5)
        self.assertEqual('Unsupported open_params "tile_size", "spatial_res"', f'{cm.exception}')

    def test_get_data_writer_ids(self):
        self.assertEqual(('*:*:memory',), self.store.get_data_writer_ids())
        self.assertEqual(('*:*:memory',), self.store.get_data_writer_ids('dataset'))
        self.assertEqual(('*:*:memory',), self.store.get_data_writer_ids('dataset[cube]'))
        self.assertEqual(('*:*:memory',), self.store.get_data_writer_ids('geodataframe'))

    def test_get_write_data_params_schema(self):
        schema = self.store.get_write_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertEqual({}, schema.properties)

    def test_write_and_delete_data(self):
        cube_3 = new_cube(variables=dict(B05=0.1, B06=0.2))
        cube_3_id = self.store.write_data(cube_3, data_id='cube_3')
        self.assertEqual('cube_3', cube_3_id)
        self.assertIs(cube_3, self.store.open_data(cube_3_id))

        cube_4 = new_cube(variables=dict(B07=0.1, B08=0.2))
        cube_4_id = self.store.write_data(cube_4)
        self.assertIsInstance(cube_4_id, str)
        self.assertIs(cube_4, self.store.open_data(cube_4_id))

        with self.assertRaises(DataStoreError) as cm:
            self.store.write_data(cube_4, tile_size=1000, spatial_res=0.5)
        self.assertEqual('Unsupported write_params "tile_size", "spatial_res"', f'{cm.exception}')

        with self.assertRaises(DataStoreError) as cm:
            self.store.write_data(cube_4, data_id='cube_3')
        self.assertEqual('Data resource "cube_3" already exist in store', f'{cm.exception}')

        self.store.delete_data(cube_3_id)
        self.store.delete_data(cube_4_id)

        with self.assertRaises(DataStoreError) as cm:
            self.store.delete_data(cube_3_id)
        self.assertEqual('Data resource "cube_3" does not exist in store', f'{cm.exception}')

        with self.assertRaises(DataStoreError) as cm:
            self.store.delete_data(cube_4_id)
        self.assertEqual(f'Data resource "{cube_4_id}" does not exist in store', f'{cm.exception}')

    def test_register_data_is_no_op(self):
        self.store.register_data('cube_3', new_cube(variables=dict(B05=0.1, B06=0.9)))
        self.assertEqual(False, self.store.has_data('cube_3'))

    def test_deregister_data_is_no_op(self):
        self.store.deregister_data('cube_1')
        self.assertEqual(True, self.store.has_data('cube_1'))
