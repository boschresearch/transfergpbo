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

from typing import List, Tuple, Callable, Dict, Hashable
from functools import partial

from emukit.core import ParameterSpace
from GPy.kern import RBF

from transfergpbo.models import (
    TaskData,
    WrapperBase,
    MHGP,
    SHGP,
    BHGP,
)
from transfergpbo.bo.run_bo import run_bo
from transfergpbo import models, benchmarks
from transfergpbo.parameters import parameters as params


def generate_functions(
    function_name: str,
    num_source_functions: int = 1,
    params_source: List[Dict[str, float]] = None,
    params_target: Dict[str, float] = None,
) -> Tuple[Callable, List[Callable], ParameterSpace]:
    """Generate the source and target functions from the respective family."""
    function = getattr(benchmarks, function_name)
    fun_target, space = (
        function() if params_target is None else function(**params_target)
    )
    funs_source = []
    for i in range(num_source_functions):
        fun, _ = function() if params_source is None else function(**params_source[i])
        funs_source.append(fun)
    return fun_target, funs_source, space


def get_benchmark(
    benchmark_name: str,
    num_source_points: List[int],
    output_noise: float = 0.0,
    params_source: List[Dict[str, float]] = None,
    params_target: Dict[str, float] = None,
) -> Tuple[Callable, Dict[Hashable, TaskData], ParameterSpace]:
    """Create the benchmark object."""
    num_source_functions = len(num_source_points)

    f_target, f_source, space = generate_functions(
        benchmark_name, num_source_functions, params_source, params_target
    )

    source_data = {}
    for i, (n_source, f) in enumerate(zip(num_source_points, f_source)):
        rand_points = space.sample_uniform(point_count=n_source)
        source_data[i] = TaskData(
            X=rand_points, Y=f(rand_points, output_noise=output_noise)
        )

    return f_target, source_data, space


def get_model(
    model_name: str, space: ParameterSpace, source_data: Dict[Hashable, TaskData]
) -> WrapperBase:
    """Create the model object."""
    model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP:
        model = model_class(space.dimensionality)
    else:
        kernel = RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    model.meta_fit(source_data)
    return model


def run_experiment(parameters: dict) -> List[float]:
    """The actual experiment code."""
    num_source_points = parameters["benchmark"]["num_source_points"]
    technique = parameters["technique"]
    benchmark_name = parameters["benchmark"]["name"]
    num_steps = parameters["benchmark"]["num_steps"]
    output_noise = parameters["output_noise"]
    params_source = parameters["benchmark"].get("parameters_source", None)
    params_target = parameters["benchmark"].get("parameters_target", None)

    # Initialize the benchmark and model
    f_target, source_data, space = get_benchmark(
        benchmark_name, num_source_points, output_noise, params_source, params_target
    )
    model = get_model(technique, space, source_data)

    # Run BO and return the regret
    return run_bo(
        experiment_fun=partial(f_target, output_noise=output_noise),
        model=model,
        space=space,
        num_iter=num_steps,
        noiseless_fun=partial(f_target, output_noise=0.0),
    )


if __name__ == "__main__":
    run_experiment(params)
