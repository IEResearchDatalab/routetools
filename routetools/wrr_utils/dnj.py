import time
import warnings
from collections.abc import Callable

import numpy as np

from routetools.wrr_bench.ocean import Ocean
from routetools.wrr_utils.route import Route
from routetools.wrr_utils.simulation import compute_times_linalg


def jacfwd(f: Callable, argnums: int = 0) -> Callable:
    """Compute the Jacobian of a function using forward differences.

    The input function f(a, b) takes two inputs, of dimension Nx2
    They represent a sequence of vectors in 2D space

    Parameters
    ----------
    f : Callable
        Function to compute the Jacobian. Assumed to take two inputs
    argnums : int, optional
        Specifies which positional argument to differentiate with
        respect to, by default 0
    """
    eps = 1e-4
    if argnums == 0:

        def jacobian(a, b):
            return np.stack(
                [
                    (f(a + np.array([eps, 0]), b) - f(a, b)) / eps,
                    (f(a + np.array([0, eps]), b) - f(a, b)) / eps,
                ],
                axis=1,
            )

    elif argnums == 1:

        def jacobian(a, b):
            return np.stack(
                [
                    (f(a, b + np.array([eps, 0])) - f(a, b)) / eps,
                    (f(a, b + np.array([0, eps])) - f(a, b)) / eps,
                ],
                axis=1,
            )

    else:
        raise ValueError("argnums must be 0 or 1")
    return jacobian


def jacrev(f: Callable, argnums: int = 0) -> Callable:
    """Compute the Jacobian of a function using reverse differences.

    The input function f(a, b) takes two inputs, of dimension Nx2
    They represent a sequence of vectors in 2D space

    Parameters
    ----------
    f : Callable
        Function to compute the Jacobian. Assumed to take two inputs
    argnums : int, optional
        Specifies which positional argument to differentiate with
        respect to, by default 0
    """
    eps = 1e-4
    if argnums == 0:

        def jacobian(a, b):
            return np.stack(
                [
                    (f(a - np.array([eps, 0]), b) - f(a, b)) / eps,
                    (f(a - np.array([0, eps]), b) - f(a, b)) / eps,
                ],
                axis=1,
            )

    elif argnums == 1:

        def jacobian(a, b):
            return np.stack(
                [
                    (f(a, b - np.array([eps, 0])) - f(a, b)) / eps,
                    (f(a, b - np.array([0, eps])) - f(a, b)) / eps,
                ],
                axis=1,
            )

    else:
        raise ValueError("argnums must be 0 or 1")
    return jacobian


def grad(f: Callable, argnums: int = 0):
    """Compute the gradient of a function.

    Parameters
    ----------
    f : Callable
        Function to compute the Jacobian
    argnums : int, optional
        Argument number to compute the Jacobian, by default 0
    """
    return jacrev(f, argnums=argnums)


def hessian(f: Callable, argnums: int = 0):
    """Compute the Hessian of a function.

    The input function f(a, b) takes two inputs, of dimension Nx2
    They represent a sequence of vectors in 2D space

    Parameters
    ----------
    f : Callable
        Function to compute the Jacobian
    argnums : int, optional
        Argument number to compute the Jacobian, by default 0
    """
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)


class DNJ:
    """Discrete Newton-Jacobi (DNJ) optimizer implementation.

    Provides an implementation of the Ferraro-Martín de Diego-Almagro
    algorithm used to locally optimize polyline waypoints.
    """

    def __init__(
        self,
        num_iter: int = 100,
        damping: float = 0.5,
        threshold: float = 0.1,
        early_stop: int = 5,
    ):
        """Initialize the Ferraro-Martín de Diego-Almagro algorithm.

        Also known as the Discrete Newton-Jacobi (DNJ).

        Parameters
        ----------
        num_iter : int, optional
            Number of DNJ iterations, by default 100
        damping : float, optional
            Similar to a 'learning rate' controls how strongly the points
            are moved towards the optimal solution, by default 0.5
        threshold : float, optional
            Maximum value that a point can be moved in a single iteration,
            in degrees, by default 0.1
        """
        self.num_iter = num_iter
        if (damping >= 1) or (damping < 0):
            raise ValueError("Damping must be between 0 and 1")
        self.damping = damping
        self.threshold = threshold
        self.early_stop = early_stop

    def optimize(self, route: Route):
        """Optimize a route for any number of iterations.

        Parameters
        ----------
        route : Route
            Route to optimize
        """
        self.comp_time = time.time()
        pts = np.stack((route.lons, route.lats), axis=1)  # degrees
        ts = route.time_stamps[1:-1]
        h = route.time_per_segment[:-1]  # Time step in seconds
        if h.min() == 0:
            raise ValueError("Time step between route waypoints cannot be zero")
        data: Ocean = route.ocean_data

        def cost_function_discretized(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
            lon0, lat0 = q0[:, 0], q0[:, 1]
            lon1, lat1 = q1[:, 0], q1[:, 1]

            cost = (
                compute_times_linalg(
                    lat0,
                    lon0,
                    lat1,
                    lon1,
                    ts,
                    route.vel_ship[:-1],
                    route.ocean_data,
                )
                ** 2
            )

            return cost

        d1ld = grad(cost_function_discretized, argnums=0)
        d2ld = grad(cost_function_discretized, argnums=1)
        d11ld = hessian(cost_function_discretized, argnums=0)
        d22ld = hessian(cost_function_discretized, argnums=1)

        def optimize_points(points: np.ndarray) -> np.ndarray:
            points_old = np.copy(points)  # degrees
            qkm1 = points[:-2]
            qk = points[1:-1]
            qkp1 = points[2:]
            b = -d2ld(qkm1, qk) - d1ld(qk, qkp1)
            a = d22ld(qkm1, qk) + d11ld(qk, qkp1)
            q = np.linalg.solve(a, b)
            # No q can be higher than threshold (either positive or negative)
            q = np.clip(q, -self.threshold, self.threshold)
            points[1:-1] = (1 - self.damping) * q + points[1:-1]
            # Locate points moved to land and roll them back to previous pos
            is_land = data.get_land_edge(points[:, 1], points[:, 0]).astype(bool)
            points[is_land] = points_old[is_land]
            return points

        # Initialize the variables
        nfail = 0
        t_best = cost_function_discretized(pts[:-2], pts[1:-1]).sum()
        pts_best = np.copy(pts)

        # Check if the points are on land and warn the user before DNJ starts
        is_land = data.get_land(route.lats, route.lons).astype(bool)
        if is_land.any():
            msg = f"[Pre-DNJ] Points cross land: {is_land.sum()} out of {len(is_land)}"
            warnings.warn(msg, stacklevel=2)

        # Loop iterations
        self.early_stop_iter = self.num_iter
        for niter in range(self.num_iter):
            pts = optimize_points(pts)
            # Compute the cost function to check if the optimization is improving
            t = cost_function_discretized(pts[:-2], pts[1:-1]).sum()
            if t > t_best:
                nfail += 1
                if nfail >= self.early_stop:
                    self.early_stop_iter = niter
                    print(f"DNJ early stopped at iteration {niter}")
                    break
            else:
                # Reset the fail counter and update the best solution
                nfail = 0
                t_best = t
                pts_best = np.copy(pts)
        # Update the points of the route, converting back to degrees
        lons, lats = pts_best[:, 0], pts_best[:, 1]

        # Check if the points are on land and warn the user
        is_land_new = data.get_land(lats, lons).astype(bool)
        if is_land_new.sum() > is_land.sum():
            msg = f"[Post-DNJ] New points are crossing land: {is_land_new.sum()}"
            warnings.warn(msg, stacklevel=2)

        route = Route.from_start_time(
            lats=lats,
            lons=lons,
            time_start=route.time_stamps[0],
            ocean_data=data,
            vel_ship=route.vel_ship,
            land_penalization=0,
        )

        self.comp_time = time.time() - self.comp_time

        return route

    def last_optimization_summary(self):
        """Return a summary of the last optimization.

        Returns
        -------
        dict
            Summary of the last optimization containing:
            - num_iter
            - damping
            - threshold
            - early_stop_threshold
            - early_stop_iter
            - comp_time
        """
        return {
            "num_iter": self.num_iter,
            "damping": self.damping,
            "threshold": self.threshold,
            "early_stop_threshold": self.early_stop,
            # Number of iterations to stop early
            "early_stop_iter": self.early_stop_iter,
            # Iteration where the optimization stopped
            "comp_time": self.comp_time,
        }
