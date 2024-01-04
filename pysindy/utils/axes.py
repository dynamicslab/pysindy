import copy
import warnings
from typing import Collection
from typing import List
from typing import NewType
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin

HANDLED_FUNCTIONS = {}

AxesWarning = type("AxesWarning", (SyntaxWarning,), {})
BasicIndexer = Union[slice, int, type(Ellipsis), type(None)]
Indexer = BasicIndexer | NDArray
StandardIndexer = Union[slice, int, type(None), NDArray]
OldIndex = NewType("OldIndex", int)  # Before moving advanced axes adajent
KeyIndex = NewType("KeyIndex", int)
NewIndex = NewType("NewIndex", int)
PartialReIndexer = tuple[KeyIndex, Optional[OldIndex], str]
CompleteReIndexer = tuple[
    list[KeyIndex], Optional[list[OldIndex]], Optional[list[NewIndex]]
]


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

    # TODO: delete default kwarg value
    def insert_axis(self, axis: Union[Collection[int], int], new_name: str = "ax_unk"):
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
            return self.axes[name]
        if parts[0] == "n":
            fwd_map = self._ax_map.fwd_map
            shape = tuple(self.shape[ax_id] for ax_id in fwd_map["ax_" + parts[1]])
            if len(shape) == 1:
                return shape[0]
            return shape
        raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

    def __getitem__(self, key: Indexer | Sequence[Indexer], /):
        output = super().__getitem__(key)
        if not isinstance(output, AxesArray):
            return output
        in_dim = self.shape
        key, adv_inds = standardize_indexer(self, key)
        adjacent, bcast_nd, bcast_start_ax = _determine_adv_broadcasting(key, adv_inds)
        remove_axes, new_axes = _apply_basic_indexing(key)

        # Handle moving around non-adjacent advanced axes
        old_index = OldIndex(0)
        pindexers: list[PartialReIndexer | list[PartialReIndexer]] = []
        for key_ind, indexer in enumerate(key):
            if isinstance(indexer, int | slice | np.ndarray):
                pindexers.append((key_ind, old_index, indexer))
                old_index += 1
            elif indexer is None:
                pindexers.append((key_ind, [None], None))
            else:
                raise TypeError(
                    f"AxesArray indexer of type {type(indexer)} not understood"
                )
        # Advanced indexing can move axes if they are not adjacent
        if not adjacent:
            _move_idxs_to_front(key, adv_inds)
            adv_inds = range(len(adv_inds))
        pindexers = _squeeze_to_sublist(pindexers, adv_inds)
        cindexers: list[CompleteReIndexer] = []
        curr_axis = 0
        for pindexer in enumerate(pindexers):
            if isinstance(pindexer, list):  # advanced indexing bundle
                bcast_idxers = _adv_broadcast_magic(key, adv_inds, pindexer)
                cindexers += bcast_idxers
                curr_axis += bcast_nd
            elif pindexer[-1] is None:
                cindexers.append((*pindexer[:-1], curr_axis))
                curr_axis += 1
            elif isinstance(pindexer[-1], int):
                cindexers.append((*pindexer[:-1], None))
            elif isinstance(pindexer[-1], slice):
                cindexers.append((*pindexer[:-1], curr_axis))
                curr_axis += 1

        if adv_inds:
            adv_inds = sorted(adv_inds)
            source_axis = [  # after basic indexing applied  # noqa
                len([id for id in range(idx_id) if key[id] is not None])
                for idx_id in adv_inds
            ]
            adv_indexers = [np.array(key[i]) for i in adv_inds]  # noqa
            bcast_nd = np.broadcast(*adv_indexers).nd
            adjacent = all(i + 1 == j for i, j in zip(adv_inds[:-1], adv_inds[1:]))
            bcast_start_ax = 0 if not adjacent else min(adv_inds)
            adv_map = {}

            for idx_id, idxer in zip(adv_inds, adv_indexers):
                base_idxer_ax_name = self._reverse_map[  # count non-None keys
                    len([id for id in range(idx_id) if key[id] is not None])
                ]
                adv_map[base_idxer_ax_name] = [
                    bcast_start_ax + shp
                    for shp in _compare_bcast_shapes(bcast_nd, idxer.shape)
                ]

            conflicts = {}
            for bcast_ax in range(bcast_nd):
                ax_names = [name for name, axes in adv_map.items() if bcast_ax in axes]
                if len(ax_names) > 1:
                    conflicts[bcast_ax] = ax_names
                    []
                if len(ax_names) == 0:
                    if "ax_unk" not in adv_map.keys():
                        adv_map["ax_unk"] = [bcast_ax + bcast_start_ax]
                    else:
                        adv_map["ax_unk"].append(bcast_ax + bcast_start_ax)

            for conflict_axis, conflict_names in conflicts.items():
                new_name = "ax_"
                for name in conflict_names:
                    adv_map[name].remove(conflict_axis)
                    if not adv_map[name]:
                        adv_map.pop(name)
                    new_name += name[3:]
                adv_map[new_name] = [conflict_axis]

            # check if integer or boolean indexing
            # if integer, check which dimensions get broadcast where
            # if multiple, axes are merged.  If adjacent, merged inplace,
            #  otherwise moved to beginning
            remove_axes.append(adv_map.keys())  # Error: remove_axis takes ints

            out_obj = np.broadcast(np.array(key[i]) for i in adv_inds)  # noqa
            pass
        # mulligan structured arrays, etc.
        new_map = _AxisMapping(
            self._ax_map.remove_axis(remove_axes), len(in_dim) - len(remove_axes)
        )
        new_map = _AxisMapping(
            new_map.insert_axis(new_axes),
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
) -> tuple[tuple[StandardIndexer], tuple[KeyIndex]]:
    """Convert any legal numpy indexer to a "standard" form.

    Standard form involves creating an equivalent indexer that is a tuple with
    one element per index of the original axis.  All advanced indexer elements
    are converted to numpy arrays
    Returns:
        A tuple of the normalized indexer as well as the indexes of
        advanced indexers
    """
    if isinstance(key, tuple):
        key = list(key)
    else:
        key = [
            key,
        ]
    if not any(ax_key is Ellipsis for ax_key in key):
        key = [*key, Ellipsis]

    _expand_indexer_ellipsis(key, arr.ndim)

    new_key: list[Indexer] = []
    adv_inds: list[int] = []
    for indexer_ind, ax_key in enumerate(key):
        if not isinstance(ax_key, BasicIndexer):
            ax_key = np.array(ax_key)
            adv_inds.append(indexer_ind)
        new_key.append(ax_key)
    return tuple(new_key), tuple(adv_inds)


def _expand_indexer_ellipsis(indexers: list[Indexer], ndim: int) -> None:
    """Replace ellipsis in indexers with the appropriate amount of slice(None)

    Mutates indexers
    """
    try:
        ellind = indexers.index(Ellipsis)
    except ValueError:
        return
    n_new_dims = sum(k is None for k in indexers)
    n_ellipsis_dims = ndim - (len(indexers) - n_new_dims - 1)
    indexers[ellind : ellind + 1] = n_ellipsis_dims * (slice(None),)


def _adv_broadcast_magic(*args):
    raise NotImplementedError


def _compare_bcast_shapes(result_ndim: int, base_shape: tuple[int]) -> list[int]:
    """Identify which broadcast shape axes are due to base_shape

    Args:
        result_ndim: number of dimensions broadcast shape has
        base_shape: shape of one element of broadcasting

    Result:
        tuple of axes in broadcast result that come from base shape
    """
    return [
        result_ndim - 1 - ax_id
        for ax_id, length in enumerate(reversed(base_shape))
        if length > 1
    ]


def _move_idxs_to_front(li: list, idxs: Sequence) -> None:
    """Move all items at indexes specified to the front of a list"""
    front = []
    for idx in reversed(idxs):
        obj = li.pop(idx)
        front.insert(0, obj)
    li = front + li


def _determine_adv_broadcasting(
    key: StandardIndexer | Sequence[StandardIndexer], adv_inds: Sequence[OldIndex]
) -> tuple:
    """Calculate the shape and location for the result of advanced indexing."""
    adjacent = all(i + 1 == j for i, j in zip(adv_inds[:-1], adv_inds[1:]))
    adv_indexers = [np.array(key[i]) for i in adv_inds]
    bcast_nd = np.broadcast(*adv_indexers).nd
    bcast_start_axis = 0 if not adjacent else min(adv_inds) if adv_inds else None
    return adjacent, bcast_nd, bcast_start_axis


def _squeeze_to_sublist(li: list, idxs: Sequence[int]) -> list:
    """Turn contiguous elements of a list into a sub-list in the same position

    e.g. _squeeze_to_sublist(["a", "b", "c", "d"], [1,2]) = ["a", ["b", "c"], "d"]
    """
    for left, right in zip(idxs[:-1], idxs[1:]):
        if left + 1 != right:
            raise ValueError("Indexes to squeeze must be contiguous")
    if not idxs:
        return li
    return li[: min(idxs)] + [[li[idx] for idx in idxs]] + li[max(idxs) + 1 :]


def _apply_basic_indexing(key: tuple[StandardIndexer]) -> tuple[list[int], list[int]]:
    """Determine where axes should be removed and added

    Only considers the basic indexers in key.  Numpy arrays are treated as
    slices, in that they don't affect the final dimensions of the output
    """
    remove_axes = []
    new_axes = []
    deleted_to_left = 0
    added_to_left = 0
    for key_ind, indexer in enumerate(key):
        if isinstance(indexer, int):
            orig_arr_axis = key_ind - added_to_left
            remove_axes.append(orig_arr_axis)
            deleted_to_left += 1
        elif indexer is None:
            new_arr_axis = key_ind - deleted_to_left
            new_axes.append(new_arr_axis)
            added_to_left += 1
    return remove_axes, new_axes


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
