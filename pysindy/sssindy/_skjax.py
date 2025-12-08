from jax import tree_util


def register_scikit_pytree(
    cls: type,
    data_fields: list[str],
    data_fit_fields: list[str],
    meta_fields: list[str],
    meta_fit_fields: list[str],
) -> type:
    """Register sklearn.BaseEstimator-like classes as pytrees

    Args:
        cls: class to decorate
        data_fields: initialization attributes that are compilable jax types
            (e.g. float, jax.Array, pytree)
        data_fit_fields: data-dependent attributes that are set with a call to fit().
            These must also be compilable jax types.
        meta_fields: initialization non-jax attributes, which must be hashable
            in order to serve as a JIT compiler cache key
        meta_fit_fields: data-dependent non-jax attributes.

    Adapted from https://github.com/jax-ml/jax/issues/25760
    """
    expected_fields = set(data_fields + meta_fields)
    total_fields = expected_fields.union(set(data_fit_fields + meta_fit_fields))

    def flatten_with_keys(obj):
        try:
            actual_fields = obj.__dict__.keys()
        except AttributeError:
            # All Python objects without __dict__ have __slots__.
            # __slots__ may be a str or iterable of strings:
            # https://docs.python.org/3/reference/datamodel.html#slots
            slots = obj.__slots__
            actual_fields = {slots} if isinstance(slots, str) else set(slots)

        if actual_fields != expected_fields and actual_fields != total_fields:
            raise TypeError(
                "unexpected attributes on object: "
                f"got {sorted(actual_fields)}, expected {sorted(expected_fields)}"
                f" or {sorted(total_fields)}"
            )

        children_with_keys = [
            (tree_util.GetAttrKey(k), getattr(obj, k)) for k in data_fields
        ]
        if data_fit_fields and hasattr(obj, data_fit_fields[0]):
            children_with_keys += [
                (tree_util.GetAttrKey(k), getattr(obj, k)) for k in data_fit_fields
            ]
        aux_data = tuple((k, getattr(obj, k)) for k in meta_fields)
        if meta_fit_fields and hasattr(obj, meta_fit_fields[0]):
            aux_data = aux_data + tuple((k, getattr(obj, k)) for k in meta_fit_fields)
        return children_with_keys, aux_data

    def unflatten_func(aux_data, children):
        result = object.__new__(cls)
        # zip will truncate to shortest, so if fit fields are not present,
        # those keys are ignored.
        for k, v in zip(data_fields + data_fit_fields, children):
            object.__setattr__(result, k, v)
        for k, v in aux_data:
            object.__setattr__(result, k, v)
        return result

    tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten_func)
    return cls
