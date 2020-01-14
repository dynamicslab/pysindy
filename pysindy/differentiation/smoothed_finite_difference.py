from scipy.signal import savgol_filter

from pysindy.differentiation import FiniteDifference


class SmoothedFiniteDifference(FiniteDifference):
    """
    Smoothed finite difference derivatives.

    Perform differentiation by smoothing input data then applying a finite
    difference method.

    Parameters
    ----------
    smoother: function, optional (default savgol_filter)
        Function to perform smoothing. Must be compatible with the
        following call signature:
        x_smoothed = smoother(x, **smoother_kws)

    smoother_kws: dict, optional (default {})
        Arguments passed to smoother when it is invoked.

    **kwargs: kwargs
        Addtional parameters passed to the FiniteDifference __init__
        function.
    """

    def __init__(self, smoother=savgol_filter, smoother_kws={}, **kwargs):
        super(SmoothedFiniteDifference, self).__init__(**kwargs)
        self.smoother = smoother
        self.smoother_kws = smoother_kws

        if smoother is savgol_filter:
            if "window_length" not in smoother_kws:
                self.smoother_kws["window_length"] = 11
            if "polyorder" not in smoother_kws:
                self.smoother_kws["polyorder"] = 3
            self.smoother_kws["axis"] = 0

    def _differentiate(self, x, t):
        """
        Apply finite difference method after smoothing.
        """
        x = self.smoother(x, **self.smoother_kws)
        return super(SmoothedFiniteDifference, self)._differentiate(x, t)
