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


from typing import List, Any, Mapping, Callable

from .undefined import UNDEFINED

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


def to_ext_obj_factory(factory_function: Callable[[], Any]):
    """
    Utility that converts a *factory_function* into a
    a extension object factory that has a "load()" method that calls the *factory_function.

    :param factory_function: no-arg function that yields the extension object
    :return: extension object factory instance
    """

    class ExtObjFactory:
        def load(self):
            return factory_function()

    return ExtObjFactory()


class Extension:
    """
    An extension in a :class:ExtensionRegistry.

    :param registry: extension registry that will host this extension
    :param type: extension type
    :param name: extension name
    :param obj: extension object
    :param metadata: extension metadata
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, registry: 'ExtensionRegistry', type: str, name: str, obj: Any, **metadata):
        loader = None
        if hasattr(obj, 'load') and callable(obj.load):
            loader = obj
            obj = UNDEFINED
        self._registry = registry
        self._type = type
        self._name = name
        self._metadata = metadata
        self._loader = loader
        self._obj = obj
        self._deleted = False

    @property
    def type(self) -> str:
        """Type of the extension."""
        return self._type

    @property
    def name(self) -> str:
        """Name of the extension."""
        return self._name

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Metadata of the extension."""
        return dict(self._metadata)

    @property
    def obj(self) -> Any:
        """The actual object instance."""
        if self._obj is UNDEFINED and self._loader is not None:
            self._obj = self._loader.load()
        return self._obj

    @property
    def deleted(self) -> bool:
        """Whether the object registration is deleted."""
        return self._deleted

    def delete(self):
        """Delete the object registration from the object registry."""
        if not self._deleted:
            # noinspection PyProtectedMember
            self._registry._delete(type=self._type, name=self._name)
            self._deleted = True


# noinspection PyShadowingBuiltins
class ExtensionRegistry:
    """
    A registry of extensions.
    Typically used by plugins to register extensions.
    """

    def __init__(self):
        self._type_to_ext = {}

    def has_ext(self, type: str, name: str) -> bool:
        """
        Test if an extension with given *type* and *name* is registered.

        :param type: extension type
        :param name: extension name
        :return: True, if extension exists
        """
        return type in self._type_to_ext and name in self._type_to_ext[type]

    def get_ext(self, type: str, name: str) -> Any:
        """
        Get registered extension for given *type* and *name*.

        :param type: extension type
        :param name: extension name
        :return: the extension
        """
        name_to_ext = self._type_to_ext[type]
        return name_to_ext[name]

    def get_ext_obj(self, type: str, name: str) -> Any:
        """
        Get extension object for given *type* and *name*.

        :param type: extension type
        :param name: extension name
        :return: extension objects
        """
        return self.get_ext(type, name).obj

    def get_all_ext_obj(self, type: str) -> List[Any]:
        """
        Get all registered extensions objects for *type*.

        :param type: extension type
        :return: list of extension objects
        """
        if type not in self._type_to_ext:
            return []
        name_to_ext = self._type_to_ext[type]
        return [ext.obj for ext in name_to_ext.values()]

    def find_ext(self,
                 type: str,
                 predicate: Callable[[Extension], bool] = None) -> List[Extension]:
        """
        Find extensions for *type* and optional filter function *predicate*.

        The filter function is called with an extension and should return
        a truth value to indicate a match or mismatch.

        :param type: extension type
        :param predicate: optional filter function
        :return: list of matching extensions
        """
        if type not in self._type_to_ext:
            return []
        name_to_ext = self._type_to_ext[type]
        if predicate is None:
            return list(name_to_ext.values())
        return [ext for ext in name_to_ext.values() if predicate(ext)]

    def add_ext_factory(self, type: str, name: str, obj_func: Callable[[], Any], **metadata) -> Extension:
        """
        Register function *obj_func* for given *type* and *name*, with optional *metadata*.

        The given *obj_func* is called to produce the actual extension object lazily.

        :param type: extension type
        :param name: extension name
        :param obj_func: extension object factory function
        :param metadata: extension metadata
        :return: a registered extension
        """
        return self.add_ext(type, name, to_ext_obj_factory(obj_func), **metadata)

    def add_ext(self, type: str, name: str, obj: Any, **metadata) -> Extension:
        """
        Register extension object *obj* for given *type* and *name*, with optional *metadata*.

        The given *obj* may be an extension object factory. In this case, *obj* has a callable
        attribute ``load`` that can be called without arguments and will return the actual extension object.
        Extension object factories are lazily, that is, only when the actual extension object is required.

        :param type: extension type
        :param name: extension name
        :param obj: extension object or extension object factory
        :param metadata: extension metadata
        :return: a registered extension
        """
        if type in self._type_to_ext:
            name_to_ext = self._type_to_ext[type]
        else:
            name_to_ext = {}
            self._type_to_ext[type] = name_to_ext
        ext = Extension(self, type, name, obj, **metadata)
        name_to_ext[name] = ext
        return ext

    def remove_ext(self, type: str, name: str):
        get_ext()
        name_to_ext = self._type_to_ext[type]
        del name_to_ext[name]
        if not name_to_ext:
            del self._type_to_ext[type]

    def _delete(self, type: str, name: str):
        name_to_ext = self._type_to_ext[type]
        del name_to_ext[name]
        if not name_to_ext:
            del self._type_to_ext[type]


_EXTENSION_REGISTRY_SINGLETON = ExtensionRegistry()


def get_ext_registry() -> ExtensionRegistry:
    """Return the extension registry singleton."""
    return _EXTENSION_REGISTRY_SINGLETON
