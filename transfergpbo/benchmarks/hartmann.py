# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from functools import partial
from typing import Tuple, Callable

import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter


def _hartmann_function(
    x: np.ndarray, alpha: np.ndarray, A: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """The hartmann function.

    Parameters
    ----------
    x:
        Numerical representation of the points for which the function should be
        evaluated. shape = (n_points, n_features)
    alpha:
        The parameters of the Hartmann function. shape = (4,)
    A:
        The A-matrix, see function documentation. shape = (4, n_features)
    P:
        The P-matrix, see function documentation. shape = (4, n_features)

    Returns
    -------
    y:
        The function value. shape = (n_points,)

    """
    exponent = np.exp(
        -np.sum(A[:, :, None] * (x.T[None, :, :] - P[:, :, None]) ** 2, axis=1)
    )
    y = (-alpha[None, :] @ exponent).reshape(-1, 1)
    return y


def hartmann3d_function(
    x: np.ndarray,
    alpha1: float = 1.0,
    alpha2: float = 1.2,
    alpha3: float = 3.0,
    alpha4: float = 3.2,
    output_noise: float = 0.0,
) -> np.ndarray:
    alpha = np.array([alpha1, alpha2, alpha3, alpha4])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )
    y = _hartmann_function(x, alpha, A, P)
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    return y


def hartmann3d(
    alpha1: float = None,
    alpha2: float = None,
    alpha3: float = None,
    alpha4: float = None,
) -> Tuple[Callable, ParameterSpace]:
    if alpha1 is None:
        alpha1 = np.random.uniform(low=1.0, high=1.02)
    if alpha2 is None:
        alpha2 = np.random.uniform(low=1.18, high=1.2)
    if alpha3 is None:
        alpha3 = np.random.uniform(low=2.8, high=3.0)
    if alpha4 is None:
        alpha4 = np.random.uniform(low=3.2, high=3.4)

    return partial(
        hartmann3d_function, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4
    ), ParameterSpace(
        [
            ContinuousParameter("x1", 0.0, 1.0),
            ContinuousParameter("x2", 0.0, 1.0),
            ContinuousParameter("x3", 0.0, 1.0),
        ]
    )


def hartmann6d_function(
    x: np.ndarray,
    alpha1: float = 1.0,
    alpha2: float = 1.2,
    alpha3: float = 3.0,
    alpha4: float = 3.2,
    output_noise: float = 0.0,
):
    alpha = np.array([alpha1, alpha2, alpha3, alpha4])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = _hartmann_function(x, alpha, A, P)
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    return y


def hartmann6d(
    alpha1: float = None,
    alpha2: float = None,
    alpha3: float = None,
    alpha4: float = None,
) -> Tuple[Callable, ParameterSpace]:
    if alpha1 is None:
        alpha1 = np.random.uniform(low=1.0, high=1.02)
    if alpha2 is None:
        alpha2 = np.random.uniform(low=1.18, high=1.2)
    if alpha3 is None:
        alpha3 = np.random.uniform(low=2.8, high=3.0)
    if alpha4 is None:
        alpha4 = np.random.uniform(low=3.2, high=3.4)

    return partial(
        hartmann6d_function, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4
    ), ParameterSpace(
        [
            ContinuousParameter("x1", 0.0, 1.0),
            ContinuousParameter("x2", 0.0, 1.0),
            ContinuousParameter("x3", 0.0, 1.0),
            ContinuousParameter("x4", 0.0, 1.0),
            ContinuousParameter("x5", 0.0, 1.0),
            ContinuousParameter("x6", 0.0, 1.0),
        ]
    )
