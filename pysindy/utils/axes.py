import copy
import warnings
from enum import Enum
from typing import Collection
from typing import List
from typing import Literal
from typing import NewType
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

HANDLED_FUNCTIONS = {}

AxesWarning = type("AxesWarning", (SyntaxWarning,), {})
BasicIndexer = Union[slice, int, type(Ellipsis), type(None), str]
Indexer = BasicIndexer | NDArray | list
StandardIndexer = Union[slice, int, type(None), NDArray[np.dtype(int)]]
OldIndex = NewType("OldIndex", int)  # Before moving advanced axes adajent
KeyIndex = NewType("KeyIndex", int)
NewIndex = NewType("NewIndex", int)
PartialReIndexer = tuple[KeyIndex, Optional[OldIndex], str]
CompleteReIndexer = tuple[
    list[KeyIndex], Optional[list[OldIndex]], Optional[list[NewIndex]]
]


class Sentinels(Enum):
    ADV_NAME = object()
    ADV_REMOVE = object()


Literal[Sentinels.ADV_NAME]


class _AxisMapping:
    """Convenience wrapper for a two-way map between axis names and
    indexes.
    """

    fwd_map: dict[str, list[int]]
    reverse_map: dict[int, str]

    def __init__(
        self,
        axes: dict[str, Union[int, Sequence[int]]] = None,
        in_ndim: int = 0,
    ):
        if axes is None:
            axes = {}
        self.fwd_map = {}
        self.reverse_map = {}

        def coerce_sequence(obj):
            if isinstance(obj, Sequence):
                return sorted(obj)
            return [obj]

        for ax_name, ax_ids in axes.items():
            ax_ids = coerce_sequence(ax_ids)
            self.fwd_map[ax_name] = ax_ids
            for ax_id in ax_ids:
                old_name = self.reverse_map.get(ax_id)
                if old_name is not None:
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
    def _compat_axes(
        in_dict: dict[str, Sequence[int]]
    ) -> dict[str, Union[Sequence[int], int]]:
        """Like fwd_map, but unpack single-element axis lists"""
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
        for cum_shift, orig_ax_remove in enumerate(sorted(axis)):
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

    def insert_axis(self, axis: Union[Collection[int], int], new_name: str):
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
        for cum_shift, ax in enumerate(sorted(axis)):
            if new_name in new_axes.keys():
                new_axes[new_name].append(ax)
            else:
                new_axes[new_name] = [ax]
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
        obj._ax_map = _AxisMapping(axes, in_ndim)
        return obj

    @property
    def axes(self):
        return self._ax_map.compat_axes

    @property
    def _reverse_map(self):
        return self._ax_map.reverse_map

    @property
    def shape(self):
        return super().shape

    def __getattr__(self, name):
        parts = name.split("_", 1)
        if parts[0] == "ax":
            try:
                return self.axes[name]
            except KeyError:
                raise AttributeError(f"AxesArray has no axis '{name}'")
        if parts[0] == "n":
            try:
                ax_ids = self._ax_map.fwd_map["ax_" + parts[1]]
            except KeyError:
                raise AttributeError(f"AxesArray has no axis '{name}'")
            shape = tuple(self.shape[ax_id] for ax_id in ax_ids)
            if len(shape) == 1:
                return shape[0]
            return shape
        raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

    def __getitem__(self, key: Indexer | Sequence[Indexer], /):
        if isinstance(key, list | np.ndarray):
            base_indexer = key
        else:
            base_indexer = tuple(None if isinstance(k, str) else k for k in key)
        output = super().__getitem__(base_indexer)
        if not isinstance(output, AxesArray):
            return output  # why?
        in_dim = self.shape
        key, adv_inds = standardize_indexer(self, key)
        bcast_nd, bcast_start_ax = _determine_adv_broadcasting(key, adv_inds)
        if adv_inds:
            key = replace_adv_indexers(key, adv_inds, bcast_start_ax, bcast_nd)
        remove_axes, new_axes, adv_names = _apply_indexing(key, self._reverse_map)
        new_axes = _rename_broadcast_axes(new_axes, adv_names)
        new_map = _AxisMapping(
            self._ax_map.remove_axis(remove_axes), len(in_dim) - len(remove_axes)
        )
        for new_ax_ind, new_ax_name in new_axes:
            new_map = _AxisMapping(
                new_map.insert_axis(new_ax_ind, new_ax_name),
                len(in_dim) - len(remove_axes) + len(new_axes),
            )
        output._ax_map = new_map
        return output

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
            self._ax_map = _AxisMapping({})
        # required by ravel() and view() used in numpy testing.  Also for zeros_like...
        elif all(
            (
                isinstance(obj, AxesArray),
                not hasattr(self, "__ax_map"),
                self.shape == obj.shape,
            )
        ):
            self._ax_map = _AxisMapping(obj.axes, len(obj.shape))
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
                axes = self._ax_map.remove_axis(axis=kwargs["axis"])
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


def standardize_indexer(
    arr: np.ndarray, key: Indexer | Sequence[Indexer]
) -> tuple[Sequence[StandardIndexer], tuple[KeyIndex, ...]]:
    """Convert any legal numpy indexer to a "standard" form.

    Standard form involves creating an equivalent indexer that is a tuple with
    one element per index of the original axis.  All advanced indexer elements
    are converted to numpy arrays, and boolean arrays are converted to
    integer arrays with obj.nonzero().
    Returns:
        A tuple of the normalized indexer as well as the indexes of
        advanced indexers
    """
    if isinstance(key, tuple):
        key = list(key)
    else:
        key = [key]

    if not any(ax_key is Ellipsis for ax_key in key):
        key = [*key, Ellipsis]

    new_key: list[Indexer] = []
    for ax_key in key:
        if not isinstance(ax_key, BasicIndexer):
            ax_key = np.array(ax_key)
            if ax_key.dtype == np.dtype(np.bool_):
                new_key += ax_key.nonzero()
                continue
        new_key.append(ax_key)

    new_key = _expand_indexer_ellipsis(new_key, arr.ndim)
    # Can't identify position of advanced indexers before expanding ellipses
    adv_inds: list[KeyIndex] = []
    for key_ind, ax_key in enumerate(new_key):
        if isinstance(ax_key, np.ndarray):
            adv_inds.append(KeyIndex(key_ind))

    return new_key, tuple(adv_inds)


def _expand_indexer_ellipsis(key: list[Indexer], ndim: int) -> list[Indexer]:
    """Replace ellipsis in indexers with the appropriate amount of slice(None)"""
    # [...].index errors if list contains numpy array
    ellind = [ind for ind, val in enumerate(key) if val is ...][0]
    new_key = []
    n_new_dims = sum(ax_key is None or isinstance(ax_key, str) for ax_key in key)
    n_ellipsis_dims = ndim - (len(key) - n_new_dims - 1)
    new_key = (
        key[:ellind]
        + n_ellipsis_dims
        * [
            slice(None),
        ]
        + key[ellind + 1 + n_ellipsis_dims :]
    )
    return new_key


def _determine_adv_broadcasting(
    key: Sequence[StandardIndexer], adv_inds: Sequence[OldIndex]
) -> tuple[int, Optional[KeyIndex]]:
    """Calculate the shape and location for the result of advanced indexing."""
    adjacent = all(i + 1 == j for i, j in zip(adv_inds[:-1], adv_inds[1:]))
    adv_indexers = [np.array(key[i]) for i in adv_inds]
    bcast_nd = np.broadcast(*adv_indexers).nd
    bcast_start_axis = 0 if not adjacent else min(adv_inds) if adv_inds else None
    return bcast_nd, KeyIndex(bcast_start_axis)


def _rename_broadcast_axes(
    new_axes: list[tuple[int, None | str | Literal[Sentinels.ADV_NAME]]],
    adv_names: list[str],
) -> list[tuple[int, str]]:
    """Normalize sentinel and NoneType names"""

    def _calc_bcast_name(*names: str) -> str:
        if not names:
            return ""
        if all(a == b for a, b in zip(names[1:], names[:-1])):
            return names[0]
        names = [name[3:] for name in dict.fromkeys(names)]  # ordered deduplication
        return "ax_" + "_".join(names)

    bcast_name = _calc_bcast_name(*adv_names)
    renamed_axes = []
    for ax_ind, ax_name in new_axes:
        if ax_name is None:
            renamed_axes.append((ax_ind, "ax_unk"))
        elif ax_name is Sentinels.ADV_NAME:
            renamed_axes.append((ax_ind, bcast_name))
        else:
            renamed_axes.append((ax_ind, "ax_" + ax_name))
    return renamed_axes


def replace_adv_indexers(
    key: Sequence[StandardIndexer],
    adv_inds: list[int],
    bcast_start_ax: int,
    bcast_nd: int,
) -> tuple[
    Union[None, str, int, Literal[Sentinels.ADV_NAME], Literal[Sentinels.ADV_REMOVE]],
    ...,
]:
    for adv_ind in adv_inds:
        key[adv_ind] = Sentinels.ADV_REMOVE
    key = key[:bcast_start_ax] + bcast_nd * [Sentinels.ADV_NAME] + key[bcast_start_ax:]
    return key


def _apply_indexing(
    key: tuple[StandardIndexer], reverse_map: dict[int, str]
) -> tuple[
    list[int], list[tuple[int, None | str | Literal[Sentinels.ADV_NAME]]], list[str]
]:
    """Determine where axes should be removed and added

    Only considers the basic indexers in key.  Numpy arrays are treated as
    slices, in that they don't affect the final dimensions of the output
    """
    remove_axes = []
    new_axes = []
    adv_names = []
    deleted_to_left = 0
    added_to_left = 0
    for key_ind, indexer in enumerate(key):
        if isinstance(indexer, int) or indexer is Sentinels.ADV_REMOVE:
            orig_arr_axis = key_ind - added_to_left
            if indexer is Sentinels.ADV_REMOVE:
                adv_names.append(reverse_map[orig_arr_axis])
            remove_axes.append(orig_arr_axis)
            deleted_to_left += 1
        elif (
            indexer is None or indexer is Sentinels.ADV_NAME or isinstance(indexer, str)
        ):
            new_arr_axis = key_ind - deleted_to_left
            new_axes.append((new_arr_axis, indexer))
            added_to_left += 1
    return remove_axes, new_axes, adv_names


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
