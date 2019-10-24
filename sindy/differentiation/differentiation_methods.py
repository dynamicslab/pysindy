import numpy as np


# TODO: implement smoothing
# TODO: consider allowing endpoints to be deleted.
#   This would require differentiation method to communicate with SINDy

def finite_difference(x, t=1, method='forward difference'):
    if method == 'forward difference':
        return forward_difference(x, t)
    else:
        return centered_difference(x, t)


# First order forward difference
# (and 2nd order backward difference for final point)
def forward_difference(x, t=1):

    # Check whether data is 1D
    if np.ndim(x) == 1:

        # Uniform timestep
        if np.isscalar(t):
            x_diff = (x[1:] - x[:-1]) / t
            backward_diff = np.array([(3*x[-1]/2 - 2*x[-2] + x[-3]/2) / t])
            return np.concatenate((x_diff, backward_diff))

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            x_diff = (x[1:] - x[:-1]) / t_diff
            backward_diff = np.array([(3*x[-1]/2 - 2*x[-2] + x[-3]/2) / t_diff[-1]])
            return np.concatenate((x_diff, backward_diff))

    # Otherwise assume data is 2D
    elif np.ndim(x) == 2:
        # Uniform timestep
        if np.isscalar(t):
            x_diff = (x[1:,:] - x[:-1,:]) / t
            backward_diff = ((3*x[-1,:]/2 - 2*x[-2,:] + x[-3,:]/2) / t).reshape(1, x.shape[1])
            return np.concatenate((x_diff, backward_diff), axis=0)

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            x_diff = (x[1:,:] - x[:-1,:]) / t_diff[:, None]
            backward_diff = ((3*x[-1,:]/2 - 2*x[-2,:] + x[-3,:]/2) / t_diff[-1]).reshape(1, x.shape[1])
            return np.concatenate((x_diff, backward_diff), axis=0)

    else:
        raise ValueError('x should be either 1-D or 2-D')

# Second order centered difference
# with third order forward/backward difference at endpoints.
# Warning: Sometimes has trouble with nonuniform grid spacing near boundaries
def centered_difference(x, t=1):
    
    # Check whether data is 1D
    if np.ndim(x) == 1:

        # Uniform timestep
        if np.isscalar(t):
            x_diff = (x[2:] - x[:-2]) / (2 * t)
            forward_diff = np.array([(-11/6 * x[0] + 3 * x[1] \
                - 3/2 * x[2] + x[3] / 3) / t])
            backward_diff = np.array([(11/6 * x[-1]-3 * x[-2] \
                + 3/2 * x[-3]-x[-4]/3) / t])
            return np.concatenate((forward_diff, x_diff, backward_diff))

        # Variable timestep
        else:
            t_diff = t[2:] - t[:-2]
            x_diff = (x[2:] - x[:-2]) / (t_diff)
            forward_diff = np.array([(-11/6 * x[0] + 3 * x[1] \
                - 3/2 * x[2] + x[3] / 3) / (t[1]-t[0])])
            backward_diff = np.array([(11/6 * x[-1]-3 * x[-2] \
                + 3/2 * x[-3]-x[-4]/3) / (t[-1]-t[-2])])
            return np.concatenate((forward_diff, x_diff, backward_diff))

    # Otherwise assume data is 2D
    elif np.ndim(x) == 2:

        # Uniform timestep
        if np.isscalar(t):
            x_diff = (x[2:,:] - x[:-2,:]) / (2 * t)
            forward_diff = ((-11/6 * x[0,:] + 3 * x[1,:] \
                - 3/2 * x[2,:] + x[3,:] / 3) / t).reshape(1, x.shape[1])
            backward_diff = ((11/6 * x[-1,:]-3 * x[-2,:] \
                + 3/2 * x[-3,:]-x[-4,:]/3) / t).reshape(1, x.shape[1])
            return np.concatenate((forward_diff, x_diff, backward_diff), axis=0)

        # Variable timestep
        else:
            t_diff = t[2:] - t[:-2]
            x_diff = (x[2:,:] - x[:-2,:]) / t_diff[:, None]
            forward_diff = ((-11/6 * x[0,:] + 3 * x[1,:] \
                - 3/2 * x[2,:] + x[3,:] / 3) / (t_diff[0]/2)).reshape(1, x.shape[1])
            backward_diff = ((11/6 * x[-1,:]-3 * x[-2,:] \
                + 3/2 * x[-3,:]-x[-4,:]/3) / (t_diff[-1]/2)).reshape(1, x.shape[1])
            return np.concatenate((forward_diff, x_diff, backward_diff), axis=0)
    
    else:
        raise ValueError('x should be either 1-D or 2-D')