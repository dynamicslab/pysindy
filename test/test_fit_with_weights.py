import numpy as np
from pysindy import SINDy


def _make_traj(t):
    # system: x_dot = x, y_dot = 1
    x = np.exp(t)
    y = t
    x_dot = x
    y_dot = np.ones_like(t)
    traj = np.vstack([x, y]).T
    traj_dot = np.vstack([x_dot, y_dot]).T
    return traj, traj_dot


def test_sindy_fit_with_various_sample_weights():
    t1 = np.linspace(0, 1, 50)
    t2 = np.linspace(0, 1, 30)

    traj1, traj1_dot = _make_traj(t1)
    traj2, traj2_dot = _make_traj(t2)

    x = [traj1, traj2]
    x_dot = [traj1_dot, traj2_dot]
    t = [t1, t2]

    # Variant A: per-trajectory 1D weights
    sw_a = [np.ones(len(t1)), np.ones(len(t2)) * 0.5]

    # Variant B: per-trajectory 2D weights with second dim = 1 (broadcast)
    sw_b = [np.ones((len(t1), 1)), np.ones((len(t2), 1)) * 0.5]

    # Variant C: per-trajectory full 2D weights (per-sample-per-coordinate)
    sw_c = [np.column_stack((np.ones(len(t1)), np.ones(len(t1)))),
            np.column_stack((np.ones(len(t2)) * 0.5, np.ones(len(t2)) * 0.5))]

    for sw in (None, sw_a, sw_b, sw_c):
        model = SINDy()
        # sample_weight=None should still fit correctly
        model.fit(x, t, x_dot=x_dot, sample_weight=sw)
        # Check that predicted derivatives match the true derivatives
        preds = model.predict(x)
        for pred_traj, true_traj in zip(preds, x_dot):
            # Allow a small numerical tolerance
            assert np.allclose(pred_traj, true_traj, atol=1e-2, rtol=1e-3)
