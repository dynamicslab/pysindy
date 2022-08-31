from typing import List

import numpy as np
from sklearn.base import TransformerMixin

HANDLED_FUNCTIONS = {}


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
        if axes is None:
            return obj
        obj.__dict__.update({**defaults, **axes})
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
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
