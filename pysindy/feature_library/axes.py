import warnings
from typing import List

import numpy as np

HANDLED_FUNCTIONS = {}


class AxesArray(np.ndarray):
    """A numpy-like array that keeps track of the meaning of its axes.

    Paramters:
        input_array (array-like): the data to create the array.
        axes (dict): A dictionary of any of

    Raises:
        ValueError if axes specification does not match shape of input_array
    """

    def __new__(cls, input_array, axes):
        obj = np.asarray(input_array).view(cls)
        defaults = {
            "ax_time": None,
            "n_time": 1,
            "ax_coord": None,
            "n_coord": 1,
            "ax_sample": None,
            "n_sample": 1,
            "ax_sample": None,
            "n_sample": 1,
            "ax_spatial": [],
            "n_spatial": [],
        }
        if axes is None:
            return obj
        new_dict = {**defaults, **axes}
        expected_dims = (
            (new_dict["ax_time"] is not None)
            + (new_dict["ax_coord"] is not None)
            + (new_dict["ax_sample"] is not None)
            + len(new_dict["ax_spatial"])
        )
        if expected_dims != len(obj.shape):
            warnings.warn(
                "Axes passed is missing values or incompatible with data"
                "given.  This occurs when reshaping data rather than creating"
                "a new AxesArray with determined axes.",
                type("AxesWarning", (PendingDeprecationWarning,), {}),
            )
            # raise ValueError("axes passed is incompatible with data given")
        # Since axes can be zero, cannot simply check "if axis:"
        else:
            if new_dict["ax_time"] is not None:
                new_dict["n_time"] = obj.shape[new_dict["ax_time"]]
            if new_dict["ax_coord"] is not None:
                new_dict["n_coord"] = obj.shape[new_dict["ax_coord"]]
            if new_dict["ax_sample"] is not None:
                new_dict["n_sample"] = obj.shape[new_dict["ax_sample"]]
            if new_dict["ax_spatial"]:
                new_dict["n_spatial"] = [obj.shape[ax] for ax in new_dict["ax_spatial"]]
        obj.__dict__.update(new_dict)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.ax_time = getattr(obj, "ax_time", None)
        self.n_time = getattr(obj, "n_time", 1)
        self.ax_coord = getattr(obj, "ax_coord", None)
        self.n_coord = getattr(obj, "n_coord", 1)
        self.ax_sample = getattr(obj, "ax_sample", None)
        self.n_sample = getattr(obj, "n_sample", 1)
        self.ax_spatial = getattr(obj, "ax_spatial", [])
        self.n_spatial = getattr(obj, "n_spatial", [])

    def __array_ufunc__(
        self, ufunc, method, *inputs, **kwargs
    ):  # this method is called whenever you use a ufunc
        f = {
            "reduce": ufunc.reduce,
            "accumulate": ufunc.accumulate,
            "reduceat": ufunc.reduceat,
            "outer": ufunc.outer,
            "at": ufunc.at,
            "__call__": ufunc,
        }
        output = AxesArray(
            f[method](*(i.view(np.ndarray) for i in inputs), **kwargs), self.__dict__
        )  # convert the inputs to np.ndarray to prevent recursion, call the
        # function, then cast it back as AxesArray
        output.__dict__ = self.__dict__  # carry forward AxesArray
        return output

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            arr = super(AxesArray, self).__array_function__(func, types, args, kwargs)
            if isinstance(arr, np.ndarray):
                return AxesArray(arr, axes=self.__dict__)
            elif arr is not None:
                return arr
            return
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
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
def concatenate(arrays, axis=0, out=None):
    parents = list(map(lambda obj: np.array(obj.data, copy=False, subok=False), arrays))
    return AxesArray(np.concatenate(parents, axis), axes=arrays[0].__dict__)


class DefaultShapedInputsMixin:
    """Identify the meaning of axes of x.

    Historically, different problem types, defined by different
    libraries, all used their own independent reshaping functions
    nested across a lot of conditionals based upon the feature
    library type.

    Now libraries can add the appropriate mixin to inherit the
    explicit reshaing functions for their problem type.
    """

    def comprehend_axes(self, x):
        "Determine appropriate axes meanings for x"
        axes = {}
        axes["ax_time"] = 0
        if len(x.shape) >= 2:
            axes["ax_coord"] = len(x.shape) - 1
        if len(x.shape) > 2:
            axes["ax_spatial"] = list(range(1, len(x.shape) - 1))
            warnings.warn("IDK how to handle this input data, losing axes labels")
        return axes

    def concat_sample_axis(self, x_list: List[AxesArray]):
        """Concatenate all trajectories and axes used to create samples."""
        new_arrs = []
        for x in x_list:
            sample_axes = (
                ([x.ax_time] if x.ax_time is not None else [])
                + ([x.ax_sample] if x.ax_sample is not None else [])
                + x.ax_spatial
            )
            new_axes = {"ax_sample": 0, "ax_coord": 1}
            n_samples = np.prod([x.shape[ax] for ax in sample_axes])
            arr = AxesArray(x.reshape((n_samples, x.shape[x.ax_coord])), new_axes)
            new_arrs.append(arr)
        return np.concatenate(new_arrs)


class PDEShapedInputsMixin:
    def comprehend_axes(self, x):
        axes = {}
        # Todo: remove time axis convetion when differentiation is done
        # explicitly along time axis
        axes["ax_coord"] = len(x.shape) - 1
        axes["ax_time"] = len(x.shape) - 2
        axes["ax_spatial"] = list(range(len(x.shape) - 2))
        return axes

    # def concat_sample_axis(self, x: AxesArray):
    #     raise NotImplementedError
