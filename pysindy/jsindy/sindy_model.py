import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jsindy.util import check_is_partial_data,get_collocation_points_weights,get_equations
from jsindy.trajectory_model import TrajectoryModel
from jsindy.dynamics_model import FeatureLinearModel
from jsindy.residual_functions import (
    FullDataTerm,PartialDataTerm,CollocationTerm,
    JointResidual)
from jsindy.optim import LMSolver
from textwrap import dedent

class JSINDyModel():
    def __init__(
        self,
        trajectory_model:TrajectoryModel,
        dynamics_model:FeatureLinearModel,
        optimizer:LMSolver = LMSolver(),
        feature_names: list[str] = None,
        input_orders: tuple[int, ...] = (0,),
        ode_order: int = 1,
    ):
        self.traj_model = trajectory_model
        self.dynamics_model = dynamics_model
        self.optimizer = optimizer
        input_orders = tuple(sorted(input_orders))
        assert input_orders[0] == 0
        self.input_orders = input_orders
        self.ode_order = ode_order
        self.variable_names = feature_names.copy()

        if self.input_orders ==(0,):
            self.feature_names = feature_names
        else:
            self.feature_names = (
                feature_names + 
                sum([
                    [f"({name}{"'"*k})" for name in feature_names] for k in self.input_orders[1:]
            ],[])
            )
    
    def __str__(self):
        traj_model_str = (self.traj_model.__str__())
        dynamics_model_str = (self.dynamics_model.__str__())
        optimizer_str = (self.optimizer.__str__())
        model_string = (
            f"""
            --------Trajectory Model--------
            {traj_model_str}

            --------Feature Library---------
            {dynamics_model_str}

            --------Optimizer Setup--------
            {optimizer_str}
            """
        )
        return '\n'.join(map(lambda x:x.lstrip(),model_string.__str__().split('\n')))

    def initialize_fit_full_obs(
        self,
        t,
        x,
        t_colloc = None,
        w_colloc = None,
        params = None,
    ):
        if params is None:
            params = dict()
        t_colloc,w_colloc = _setup_colloc(t,t_colloc,w_colloc)
        
        self.t_colloc = t_colloc
        self.w_colloc = w_colloc
        self.t = t
        self.x = x

        params = self.traj_model.initialize(
            self.t,self.x,t_colloc,params
            )
        
        params = self.dynamics_model.initialize(
            self.t,self.x,params,self.input_orders
        )

        self.data_term = FullDataTerm(
            self.t,self.x,self.traj_model
        )
        self.colloc_term = CollocationTerm(
            self.t_colloc,self.w_colloc,
            self.traj_model,self.dynamics_model,
            input_orders = self.input_orders,ode_order = self.ode_order
        )
        self.residuals = JointResidual(self.data_term,self.colloc_term)
        return params
    
    def initialize_fit_partial_obs(
        self,
        t,
        y,
        v,
        t_colloc = None,
        w_colloc = None,
        params= None
        ):
        if params is None:
            params = dict()
        t_colloc,w_colloc = _setup_colloc(t,t_colloc,w_colloc)

        self.t_colloc = t_colloc
        self.w_colloc = w_colloc
        self.t = t
        self.y = y
        self.v = v

        params = self.traj_model.initialize_partialobs(
            self.t,self.y,self.v,t_colloc,params
            )
        
        params = self.dynamics_model.initialize_partialobs(
            self.t,self.y,self.v,params,self.input_orders
        )
        
        self.data_term = PartialDataTerm(
            self.t,self.y,self.v,self.traj_model
        )
        self.colloc_term = CollocationTerm(
            self.t_colloc,self.w_colloc,
            self.traj_model,self.dynamics_model,
            input_orders = self.input_orders,ode_order = self.ode_order
        )
        self.residuals = JointResidual(self.data_term,self.colloc_term)
        return params
        
        
    def fit(
        self,
        t,
        x = None,
        t_colloc = None,
        w_colloc = None,
        params = None,
        partialobs_y = None,
        partialobs_v = None,
    ):
        #TODO: Add a logs dictionary that's carried around in the same way that params is
        
        if params is None:
            params = dict()
        params["show_progress"] = self.optimizer.solver_settings.show_progress

        is_partially_observed = check_is_partial_data(t,x,partialobs_y,partialobs_v)
        self.is_partially_observed = is_partially_observed
        if is_partially_observed is True:
            params = self.initialize_fit_partial_obs(
                t,partialobs_y,partialobs_v,
                t_colloc,w_colloc,params
                )
        else:
            params = self.initialize_fit_full_obs(
                t,x,t_colloc,
                w_colloc, params
                )

        z,theta,opt_result,params = self.optimizer.run(self,params)
        self.z = z
        self.theta = theta
        self.opt_result = opt_result
        self.params = params
    
    def print(self,theta=None, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        if theta is None:
            theta = self.theta
        eqns = get_equations(
            coef = theta.T,
            feature_names = self.feature_names,
            feature_library = self.dynamics_model.feature_map,
            precision = precision
            )
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(eqns))]
        else:
            feature_names = self.variable_names

        for name, eqn in zip(feature_names, eqns, strict=True):
            lhs = f"({name}){"'"*self.ode_order}"
            print(f"{lhs} = {eqn}", **kwargs)
            
    def predict(self,x,theta = None):
        if theta is None:
            theta = self.theta
        return self.dynamics_model.predict(x,theta)

    def predict_state(self,t,z = None):
        if z is None:
            z = self.z
        return self.traj_model.predict(t,z)

def _setup_colloc(t,t_colloc,w_colloc):
    if t_colloc is not None and w_colloc is None:
        w_colloc = 1/len(t_colloc) * jnp.ones_like(t_colloc)

    if t_colloc is None:
        t_colloc,w_colloc = get_collocation_points_weights(t)
    return t_colloc,w_colloc
