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

from typing import Callable, Union
import numpy as np
import scipy.optimize as opt

from emukit.core.interfaces import IModel
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core import ParameterSpace
from emukit.bayesian_optimization.acquisitions import (
    NegativeLowerConfidenceBound as UCB,
)

from transfergpbo.models import TaskData, Model


def shgo_minimize(fun: Callable, search_space: ParameterSpace) -> opt.OptimizeResult:
    """Minimize the benchmark with simplicial homology global optimization, SHGO

    Original paper: https://doi.org/10.1007/s10898-018-0645-y

    Parameters
    -----------
    fun
        The function to be minimized.
    search_space
        Fully described search space with valid bounds and a meaningful prior.

    Returns
    --------
    res
        The optimization result represented as a `OptimizeResult` object.
    """

    def objective(x):
        benchmark_value = fun(np.atleast_2d(x), output_noise=0.0)
        return benchmark_value.squeeze()

    bounds = search_space.get_bounds()
    return opt.shgo(objective, bounds=bounds, sampling_method="sobol")


def run_bo(
    experiment_fun: Callable,
    model: Union[Model, IModel],
    space: ParameterSpace,
    num_iter: int,
    noiseless_fun: Callable = None,
):
    """Runs Bayesian optimization."""

    if noiseless_fun:
        f_min = shgo_minimize(noiseless_fun, space).fun
    else:
        f_min = None

    regret = []
    for i in range(num_iter):
        print(f"Processing for step: {i + 1}")

        if i == 0:  # sample a random point for the first experiment
            X_new = space.sample_uniform(1)
            Y_new = experiment_fun(X_new)
            X, Y = X_new, Y_new
        else:  # optimize the AF
            af = UCB(model, beta=np.float64(3.0))
            optimizer = GradientAcquisitionOptimizer(space)
            X_new, _ = optimizer.optimize(af)
            Y_new = experiment_fun(X_new)
            X = np.append(X, X_new, axis=0)
            Y = np.append(Y, Y_new, axis=0)

        print(f"Next training point is: {X_new}, {Y_new}")

        model.fit(TaskData(X, Y), optimize=True)

        if f_min is not None:
            f_min_observed = np.min(experiment_fun(X, output_noise=0.0))
            regret.append((f_min_observed - f_min).item())

    print("BO loop is finished.")
    return regret
