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


def branin_function(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 0.1,
    c: float = 1.0,
    r: float = 5.0,
    s: float = 10.0,
    t: float = 0.05,
    output_noise: float = 0.0,
) -> np.ndarray:
    x1 = np.array(x[:, 0]).reshape(-1, 1)
    x2 = np.array(x[:, 1]).reshape(-1, 1)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    y = y.reshape(-1, 1)
    y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
    return y


def branin(
    a: float = None,
    b: float = None,
    c: float = None,
    r: float = None,
    s: float = None,
    t: float = None,
) -> Tuple[Callable, ParameterSpace]:
    if a is None:
        a = np.random.uniform(low=0.5, high=1.5)
    if b is None:
        b = np.random.uniform(low=0.1, high=0.15)
    if c is None:
        c = np.random.uniform(low=1.0, high=2.0)
    if r is None:
        r = np.random.uniform(low=5.0, high=7.0)
    if s is None:
        s = np.random.uniform(low=8.0, high=12.0)
    if t is None:
        t = np.random.uniform(low=0.03, high=0.05)

    return partial(branin_function, a=a, b=b, c=c, r=r, s=s, t=t), ParameterSpace(
        [ContinuousParameter("x1", -5.0, 10.0), ContinuousParameter("x2", 0.0, 15.0)]
    )
