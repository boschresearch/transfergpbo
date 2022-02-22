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

from typing import Tuple, Callable
from functools import partial

import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter


def alpine_function(x, s: float = 0.0, output_noise: float = 0.0):
    x = np.asarray(x)
    y = (x * np.sin(x + np.pi + s) + x / 10).reshape(-1, 1)
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    return y


def alpine(s: float = None) -> Tuple[Callable, ParameterSpace]:
    if s is None:
        s = np.random.uniform(low=0.0, high=np.pi / 2)
    return partial(alpine_function, s=s), ParameterSpace(
        [ContinuousParameter("x", -10, 10)]
    )
