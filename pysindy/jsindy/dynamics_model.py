from abc import ABC

import jax
import jax.numpy as jnp

import pysindy as ps
from .util import l2reg_lstsq


class DynamicsModel(ABC):
    def predict(self, x, theta):
        pass

class PolyLib(ps.PolynomialLibrary):
    def fit(self, x: jax.Array):
        #Using ps.PolynomialLibrary to get powers right now
        super().fit(x)
        self.jpowers_ = jnp.array(self.powers_)

    def transform(self, x: jax.Array):
        if jnp.ndim(x)==2:
            return jnp.prod(jax.vmap(jnp.pow,in_axes=(None,0))(x,self.jpowers_),axis=2).T
        elif jnp.ndim(x)==1:
            return jnp.prod(jax.vmap(jnp.pow,in_axes=(None,0))(x,self.jpowers_),axis=1)
        else:
            raise ValueError(f"Polynomial library cannot handle input shape, {x.shape}")

    def __call__(self, X):
        return self.transform(X)

class FeatureLinearModel(DynamicsModel):
    def __init__(
            self, 
            feature_map=PolyLib(degree=2),
            reg_scaling = 1.
            ) -> None:
        self.feature_map = feature_map
        self.attached = False
        self.reg_scaling = reg_scaling
    
    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.feature_map.get_params().items()])        

    def attach(self, x: jax.Array,input_orders = (0,)):
        shaped_features = jnp.hstack([x]*len(input_orders))
        
        self.feature_map.fit(shaped_features)
        self.num_targets = x.shape[1]

        self.num_theta = (
            self.num_targets * self.feature_map.n_output_features_
        )
        self.num_features = self.feature_map.n_output_features_
        self.attached = True
        self.regmat = self.reg_scaling*jnp.eye(self.num_theta)
        
        self.tot_params = self.num_features*self.num_targets
        self.param_shape = (self.num_features, self.num_targets)
    
    def initialize(self,t,x,params,input_orders):
        self.attach(x,input_orders = input_orders)
        return params
    
    def initialize_partialobs(self,t,y,v,params,input_orders):
        #Pretending that v is x gives all of the right shapes
        self.attach(v,input_orders = input_orders)
        return params


    # somewhere in jsindy.fit a predict is used and needs to fixed 
    def predict(self, x, theta):
        if jnp.ndim(x)==1:
            return self.feature_map.transform(x) @ theta
        elif jnp.ndim(x)==2:
            return self.feature_map.transform(x) @ theta
        else:
            raise ValueError(f"x shape not compatible, x.shape = {x.shape}")
        
    def __call__(self, x,theta):
        return self.predict(x,theta)
    
    def get_fitted_theta(self,x,xdot,lam = 1e-2):
        A = self.feature_map.transform(x)
        return l2reg_lstsq(A,xdot,reg = lam)

# class FeatureLinearModel():
#     def __init__(
#         self,
#         feature_map,
#         in_dim,
#         out_dim,
#         ):
#         self.shape = ...
#         self.feature_map = feature_map
#         self.regularization_weights = ...

#     def featurize(self,x):
#         return self.feature_map(x)
    
#     def predict(self, x,theta):
#         FX = self.featurize(x)
#         return FX@theta
    
#     def __call__(self, x,theta):
#         self.predict(x,theta)
