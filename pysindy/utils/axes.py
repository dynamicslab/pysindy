import copy
import warnings
from typing import Collection
from typing import List
from typing import Sequence

import numpy as np
from sklearn.base import TransformerMixin

HANDLED_FUNCTIONS = {}

AxesWarning = type("AxesWarning", (SyntaxWarning,), {})


class AxesArray(np.lib.mixins.NDArrayOperatorsMixin, np.ndarray):
    """A numpy-like array that keeps track of the meaning of its axes.

    Parameters:
        input_array (array-like): the data to create the array.
        axes (dict): A dictionary of axis labels to shape indices.
            Allowed keys:
            -  ax_time: int
            -  ax_coord: int
            -  ax_sample: int
            -  ax_spatial: List[int]

    Raises:
        AxesWarning if axes does not match shape of input_array
    """

    def __new__(cls, input_array, axes):
        obj = np.asarray(input_array).view(cls)
        defaults = {
            "ax_time": None,
            "ax_coord": None,
            "ax_sample": None,
            "ax_spatial": [],
        }
        n_axes = sum(1 for k, v in axes.items() if v)
        if axes is None:
            return obj
        in_ndim = len(input_array.shape)
        if n_axes != in_ndim:
            warnings.warn(
                f"{n_axes} axes labeled for array with {in_ndim} axes", AxesWarning
            )
        axes = {**defaults, **axes}
        listed_axes = [
            el for k, v in axes.items() if isinstance(v, Collection) for el in v
        ]
        listed_axes += [
            v
            for k, v in axes.items()
            if not isinstance(v, Collection) and v is not None
        ]
        _reverse_map = {}
        for axis in listed_axes:
            if axis >= in_ndim:
                raise ValueError(
                    f"Assigned definition to axis {axis}, but array only has"
                    f" {in_ndim} axes"
                )
            ax_names = [ax_name for ax_name in axes if axes[ax_name] == axis]
            if len(ax_names) > 1:
                raise ValueError(f"Assigned multiple definitions to axis {axis}")
            _reverse_map[axis] = ax_names[0]
        obj.__dict__.update({**axes})
        obj.__dict__["_reverse_map"] = _reverse_map
        return obj

    def __getitem__(self, key, /):
        remove_axes = []
        if isinstance(key, int):
            remove_axes.append(key)
        if isinstance(key, Sequence):
            for axis, k in enumerate(key):
                if isinstance(k, int):
                    remove_axes.append(axis)
        new_item = super().__getitem__(key)
        if not isinstance(new_item, AxesArray):
            return new_item
        for axis in remove_axes:
            ax_name = self._reverse_map[axis]
            if isinstance(new_item.__dict__[ax_name], int):
                new_item.__dict__[ax_name] = None
            else:
                new_item.__dict__[ax_name].remove(axis)
            new_item._reverse_map.pop(axis)
        return new_item

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self._reverse_map = copy.deepcopy(getattr(obj, "_reverse_map", {}))
        self.ax_time = getattr(obj, "ax_time", None)
        self.ax_coord = getattr(obj, "ax_coord", None)
        self.ax_sample = getattr(obj, "ax_sample", None)
        self.ax_spatial = getattr(obj, "ax_spatial", [])

    @property
    def n_spatial(self):
        return tuple(self.shape[ax] for ax in self.ax_spatial)

    @property
    def n_time(self):
        return self.shape[self.ax_time] if self.ax_time is not None else 1

    @property
    def n_sample(self):
        return self.shape[self.ax_sample] if self.ax_sample is not None else 1

    @property
    def n_coord(self):
        return self.shape[self.ax_coord] if self.ax_coord is not None else 1

    @property
    def shape(self):
        return super().shape

    def __array_ufunc__(
        self, ufunc, method, *inputs, out=None, **kwargs
    ):  # this method is called whenever you use a ufunc
        args = []
        for input_ in inputs:
            if isinstance(input_, AxesArray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, AxesArray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        if method == "at":
            return
        if ufunc.nout == 1:
            results = (results,)
        results = tuple(
            (AxesArray(np.asarray(result), self.__dict__) if output is None else output)
            for result, output in zip(results, outputs)
        )
        return results[0] if len(results) == 1 else results

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            arr = super(AxesArray, self).__array_function__(func, types, args, kwargs)
            if isinstance(arr, np.ndarray):
                return AxesArray(arr, axes=self.__dict__)
            elif arr is not None:
                return arr
            return
        if not all(issubclass(t, AxesArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(arrays, axis=0):
    parents = [np.asarray(obj) for obj in arrays]
    ax_list = [obj.__dict__ for obj in arrays if isinstance(obj, AxesArray)]
    for ax1, ax2 in zip(ax_list[:-1], ax_list[1:]):
        if ax1 != ax2:
            raise TypeError("Concatenating >1 AxesArray with incompatible axes")
    return AxesArray(np.concatenate(parents, axis), axes=ax_list[0])


def comprehend_axes(x):
    axes = {}
    axes["ax_coord"] = len(x.shape) - 1
    axes["ax_time"] = len(x.shape) - 2
    axes["ax_spatial"] = list(range(len(x.shape) - 2))
    return axes


class SampleConcatter(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x_list, y_list):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, x_list):
        return concat_sample_axis(x_list)


def concat_sample_axis(x_list: List[AxesArray]):
    """Concatenate all trajectories and axes used to create samples."""
    new_arrs = []
    for x in x_list:
        sample_axes = (
            x.ax_spatial
            + ([x.ax_time] if x.ax_time is not None else [])
            + ([x.ax_sample] if x.ax_sample is not None else [])
        )
        new_axes = {"ax_sample": 0, "ax_coord": 1}
        n_samples = np.prod([x.shape[ax] for ax in sample_axes])
        arr = AxesArray(x.reshape((n_samples, x.shape[x.ax_coord])), new_axes)
        new_arrs.append(arr)
    return np.concatenate(new_arrs, axis=new_arrs[0].ax_sample)


def wrap_axes(axes: dict, obj):
    """Add axes to object (usually, a sparse matrix)"""

    for key in ["ax_spatial", "ax_time", "ax_sample", "ax_coord"]:
        try:
            obj.__setattr__(key, axes[key])
        except KeyError:
            pass
    return obj
