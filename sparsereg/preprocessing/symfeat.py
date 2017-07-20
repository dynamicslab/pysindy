from itertools import product, chain, combinations
from collections import OrderedDict

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sympy import simplify
from toolz import compose
from joblib import hash as _hash


class Base(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.name_cache = None

    @property
    def name(self):
        if self.name_cache == None:
            self.name_cache = str(simplify(self._name))
        return self.name_cache


class SimpleFeature(Base):
    """Base to create polynomial features.
    """
    def __init__(self, exponent, index=0):
        super().__init__()
        if exponent == 0:
            raise ValueError
        self.exponent = exponent
        self.index = index

    def transform(self, x):
        return x[:, self.index]**self.exponent

    @property
    def _name(self):
        if self.exponent == 1:
            return "x_{}".format(self.index)
        else:
            return "x_{}**{}".format(self.index, self.exponent)


class OperatorFeature(Base):
    def __init__(self, feat_cls, operator, operator_name=None):
        super().__init__()
        self.feat_cls = feat_cls
        self.operator = operator
        self.operator_name = operator_name or str(operator)

    def transform(self, x):
        return self.operator(self.feat_cls.transform(x))

    @property
    def _name(self):
        return "{}({})".format(self.operator_name, self.feat_cls.name)


class ProductFeature(Base):
    def __init__(self, feat_cls1, feat_cls2):
        super().__init__()
        self.feat_cls1 = feat_cls1
        self.feat_cls2 = feat_cls2

    def transform(self, x):
        return self.feat_cls1.transform(x) * self.feat_cls2.transform(x)

    @property
    def _name(self):
        return "{}*{}".format(self.feat_cls1.name, self.feat_cls2.name)


def _allfinite(tpl):
    b, x = tpl
    return np.all(np.isfinite(x))


def _take_finite(x):
    return list(filter(_allfinite, x))


def _hash(array):
    return hash(str(array))


def hashed_hash_():
    cache = {}
    def inner(x):
        key = _hash(x)
        if key not in cache:
            cache[key] = _hash(x)
        return cache[key]
    return inner

hashed_hash = hashed_hash_()

def _remove_id(tpl):
    expr = OrderedDict()
    redundant = []
    for b, x in tpl:
        name = b.name
        vhash = hashed_hash(x)
        if name not in expr and vhash not in redundant:
            expr[name] = b, x
            redundant.append(vhash)
    return list(expr.values())

get_valid = compose(_remove_id, _take_finite)

class SymbolicFeatures(Base):
    """Main class"""
    def __init__(self, exponents, operators, consider_products=True):
        self.exponents = exponents
        self.operators = operators
        self.consider_products = consider_products
        self._precompute_hash = None
        self._names = None
        self.n_features = None

    def fit(self, x, y=None):
        x = np.asfortranarray(x)
        _, self.n_features = x.shape
        # 1) Get all simple features
        simple = (SimpleFeature(e, index=i) for e, i in product(self.exponents, range(self.n_features)))
        simple = get_valid((s, s.transform(x)) for s in simple)
        # 2) Get all operator features
        operator = (OperatorFeature(s, op, operator_name=op_name) for (s, _), (op_name, op) in product(simple, self.operators.items()))
        operator = get_valid((o, o.transform(x)) for o in operator)
        # 3) Get all product features
        all_ = simple + operator

        if self.consider_products:
            combs = chain(product(operator, simple), combinations(simple, 2))
            prod = [ProductFeature(feat1, feat2) for (feat1, _) , (feat2, _) in combs]
            all_ += get_valid((p, p.transform(x)) for p in prod)

        all_ = get_valid(all_)
        feat_cls, features = zip(*[(c, np.array(f)) for c, f in all_])

        self._precomputed_features = np.array(list(features)).T  # speed up fit_transform
        self._precompute_hash = _hash(x)

        self.feat_cls = list(feat_cls)
        return self

    def transform(self, x):
        if self._precompute_hash == _hash(x):
            return self._precomputed_features
        else:
            features = [c.transform(x) for c in self.feat_cls]
            return np.array(list(features)).T

    def get_feature_names(self, input_features=None):
        """Get all the feature names. Only Available after fitting."""
        if self._names is None:
            self._names = [f.name for f in self.feat_cls]
        if input_features:
            for i, input_feature in enumerate(input_features):
                self._names = [n.replace("x_{}".format(i), input_feature) for n in self._names]
        return self._names


    def __getstate__(self):
        state = self.__dict__.copy()
        try:
            del state["_precomputed_features"]
        except KeyError:
            pass
        state["_precompute_hash"] = None
        return state
