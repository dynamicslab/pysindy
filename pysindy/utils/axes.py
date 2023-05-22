import copy
import warnings
from typing import Collection
from typing import List
from typing import MutableMapping
from typing import Sequence
from typing import Union

import numpy as np
from sklearn.base import TransformerMixin

HANDLED_FUNCTIONS = {}

AxesWarning = type("AxesWarning", (SyntaxWarning,), {})


class _AxisMapping:
    """Convenience wrapper for a two-way map between axis names and
    indexes.
    """

    def __init__(
        self,
        axes: MutableMapping[str, Union[int, Sequence[int]]] = None,
        in_ndim: int = 0,
    ):
        if axes is None:
            axes = {}
        axes = copy.deepcopy(axes)
        self.fwd_map: dict[str, list[int]] = {}
        self.reverse_map: dict[int, str] = {}
        null = object()

        def coerce_sequence(obj):
            if isinstance(obj, Sequence):
                return sorted(obj)
            return [obj]

        for ax_name, ax_ids in axes.items():
            ax_ids = coerce_sequence(ax_ids)
            self.fwd_map[ax_name] = ax_ids
            for ax_id in ax_ids:
                old_name = self.reverse_map.get(ax_id, null)
                if old_name is not null:
                    raise ValueError(f"Assigned multiple definitions to axis {ax_id}")
                if ax_id >= in_ndim:
                    raise ValueError(
                        f"Assigned definition to axis {ax_id}, but array only has"
                        f" {in_ndim} axes"
                    )
                self.reverse_map[ax_id] = ax_name
        if len(self.reverse_map) != in_ndim:
            warnings.warn(
                f"{len(self.reverse_map)} axes labeled for array with {in_ndim} axes",
                AxesWarning,
            )

    @staticmethod
    def _compat_axes(in_dict: dict[str, Sequence]) -> dict[str, Union[Sequence, int]]:
        """Turn single-element axis index lists into ints"""
        axes = {}
        for k, v in in_dict.items():
            if len(v) == 1:
                axes[k] = v[0]
            else:
                axes[k] = v
        return axes

    @property
    def compat_axes(self):
        return self._compat_axes(self.fwd_map)

    def remove_axis(self, axis: Union[Collection[int], int, None] = None):
        """Create an axes dict from self with specified axis or axes
        removed and all greater axes decremented.

        Arguments:
            axis: the axis index or axes indexes to remove.  By numpy
            ufunc convention, axis=None (default) removes _all_ axes.
        """
        if axis is None:
            return {}
        new_axes = copy.deepcopy(self.fwd_map)
        in_ndim = len(self.reverse_map)
        if not isinstance(axis, Collection):
            axis = [axis]
        for cum_shift, orig_ax_remove in enumerate(axis):
            remove_ax_name = self.reverse_map[orig_ax_remove]
            curr_ax_remove = orig_ax_remove - cum_shift
            if len(new_axes[remove_ax_name]) == 1:
                new_axes.pop(remove_ax_name)
            else:
                new_axes[remove_ax_name].remove(curr_ax_remove)
            for old_ax_dec in range(curr_ax_remove + 1, in_ndim - cum_shift):
                orig_ax_dec = old_ax_dec + cum_shift
                ax_dec_name = self.reverse_map[orig_ax_dec]
                new_axes[ax_dec_name].remove(old_ax_dec)
                new_axes[ax_dec_name].append(old_ax_dec - 1)
        return self._compat_axes(new_axes)

    def insert_axis(self, axis: Union[Collection[int], int]):
        """Create an axes dict from self with specified axis or axes
        added and all greater axes incremented.

        Arguments:
            axis: the axis index or axes indexes to add.

        Todo:
            May be more efficient to determine final axis-to-axis
            mapping, then apply, rather than apply changes after each
            axis insert.
        """
        new_axes = copy.deepcopy(self.fwd_map)
        in_ndim = len(self.reverse_map)
        if not isinstance(axis, Collection):
            axis = [axis]
        for cum_shift, ax in enumerate(axis):
            if "ax_unk" in new_axes.keys():
                new_axes["ax_unk"].append(ax)
            else:
                new_axes["ax_unk"] = [ax]
            for ax_id in range(ax, in_ndim + cum_shift):
                ax_name = self.reverse_map[ax_id - cum_shift]
                new_axes[ax_name].remove(ax_id)
                new_axes[ax_name].append(ax_id + 1)
        return self._compat_axes(new_axes)


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
        if axes is None:
            axes = {}
        in_ndim = len(input_array.shape)
        obj.__ax_map = _AxisMapping(axes, in_ndim)
        return obj

    @property
    def axes(self):
        return self.__ax_map.compat_axes

    @property
    def _reverse_map(self):
        return self.__ax_map.reverse_map

    @property
    def shape(self):
        return super().shape

    def __getattr__(self, name):
        parts = name.split("_", 1)
        if parts[0] == "ax":
            return self.axes[name]
        if parts[0] == "n":
            fwd_map = self.__ax_map.fwd_map
            shape = tuple(self.shape[ax_id] for ax_id in fwd_map["ax_" + parts[1]])
            if len(shape) == 1:
                return shape[0]
            return shape
        raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

    def __getitem__(self, key, /):
        output = super().__getitem__(key)
        if not isinstance(output, AxesArray):
            return output
        # determine axes of output
        in_dim = self.shape  # noqa
        out_dim = output.shape  # noqa
        remove_axes = []  # noqa
        new_axes = []  # noqa
        basic_indexer = Union[slice, int, type(Ellipsis), np.newaxis, type(None)]
        if any(
            (  # basic indexing
                isinstance(key, basic_indexer),
                isinstance(key, tuple)
                and all(isinstance(k, basic_indexer) for k in key),
            )
        ):
            key = _standardize_basic_indexer(self, key)
            shift = 0
            for ax_ind, indexer in enumerate(key):
                if indexer is None:
                    new_axes.append(ax_ind - shift)
                elif isinstance(indexer, int):
                    remove_axes.append(ax_ind)
                    shift += 1
        elif any(  # fancy indexing
            (
                isinstance(key, Sequence) and not isinstance(key, tuple),
                isinstance(key, np.ndarray),
                isinstance(key, tuple) and any(isinstance(k, Sequence) for k in key),
                isinstance(key, tuple)
                and any(isinstance(k, np.ndarray) for k in key),  # ?
            )
        ):
            # check if integer or boolean indexing
            # if integer, check which dimensions get broadcast where
            # if multiple, axes are merged.  If adjacent, merged inplace,
            #  otherwise moved to beginning
            pass
        else:
            raise TypeError(f"AxisArray {self} does not know how to slice with {key}")
        # mulligan structured arrays, etc.
        new_map = _AxisMapping(
            self.__ax_map.remove_axis(remove_axes), len(in_dim) - len(remove_axes)
        )
        new_map = _AxisMapping(
            new_map.insert_axis(new_axes),
            len(in_dim) - len(remove_axes) + len(new_axes),
        )
        output.__ax_map = new_map
        return output

    # def __getitem__(self, key, /):
    #     remove_axes = []
    #     if isinstance(key, int):
    #         remove_axes.append(key)
    #     if isinstance(key, Sequence):
    #         for axis, k in enumerate(key):
    #             if isinstance(k, int):
    #                 remove_axes.append(axis)
    #     new_item = super().__getitem__(key)
    #     if not isinstance(new_item, AxesArray):
    #         return new_item
    #     for axis in remove_axes:
    #         ax_name = self._reverse_map[axis]
    #         if isinstance(new_item.__dict__[ax_name], int):
    #             new_item.__dict__[ax_name] = None
    #         else:
    #             new_item.__dict__[ax_name].remove(axis)
    #         new_item._reverse_map.pop(axis)
    #     return new_item

    def __array_wrap__(self, out_arr, context=None):
        return super().__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj) -> None:
        if obj is None:  # explicit construction via super().__new__().. not called?
            return
        # view from numpy array, called in constructor but also tests
        if all(
            (
                not isinstance(obj, AxesArray),
                self.shape == (),
                not hasattr(self, "__ax_map"),
            )
        ):
            self.__ax_map = _AxisMapping({})
        # required by ravel() and view() used in numpy testing.  Also for zeros_like...
        elif all(
            (
                isinstance(obj, AxesArray),
                not hasattr(self, "__ax_map"),
                self.shape == obj.shape,
            )
        ):
            self.__ax_map = _AxisMapping(obj.axes, len(obj.shape))
        # maybe add errors for incompatible views?

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
        if method == "reduce" and (
            "keepdims" not in kwargs.keys() or kwargs["keepdims"] is False
        ):
            axes = None
            if kwargs["axis"] is not None:
                axes = self.__ax_map.remove_axis(axis=kwargs["axis"])
        else:
            axes = self.axes
        final_results = []
        for result, output in zip(results, outputs):
            if output is not None:
                final_results.append(output)
            elif axes is None:
                final_results.append(result)
            else:
                final_results.append(AxesArray(np.asarray(result), axes))
        results = tuple(final_results)
        return results[0] if len(results) == 1 else results

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            arr = super(AxesArray, self).__array_function__(func, types, args, kwargs)
            if isinstance(arr, AxesArray):
                return arr
            elif isinstance(arr, np.ndarray):
                return AxesArray(arr, axes=self.axes)
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
    ax_list = [obj.axes for obj in arrays if isinstance(obj, AxesArray)]
    for ax1, ax2 in zip(ax_list[:-1], ax_list[1:]):
        if ax1 != ax2:
            raise TypeError("Concatenating >1 AxesArray with incompatible axes")
    return AxesArray(np.concatenate(parents, axis), axes=ax_list[0])


def _standardize_basic_indexer(arr: np.ndarray, key):
    """Convert to a tuple of slices, ints, and None."""
    if isinstance(key, tuple):
        if not any(ax_key is Ellipsis for ax_key in key):
            key = (*key, Ellipsis)
        slicedim = sum(isinstance(ax_key, slice | int) for ax_key in key)
        final_key = []
        for ax_key in key:
            inner_iterator = (ax_key,)
            if ax_key is Ellipsis:
                inner_iterator = (arr.ndim - slicedim) * (slice(None),)
            for el in inner_iterator:
                final_key.append(el)
        return tuple(final_key)
    return _standardize_basic_indexer(arr, (key,))


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
