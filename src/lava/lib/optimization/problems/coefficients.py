# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

import numpy as np
import numpy.typing as npt


class CoefficientTensorsMixin:
    def __init__(self,
                 *coefficients: ty.Union[ty.List, npt.ArrayLike]):
        """Coefficients for a scalar function of a vector.

        Parameters
        ----------
        kwargs:  the tensor coefficients of the function.
        """
        c_dict = dict()
        for coefficient in coefficients:
            if type(coefficient) in [list, int]:
                coefficient = np.asarray(coefficient)
                rank = coefficient.ndim
            elif type(coefficient) is not np.ndarray:
                raise ValueError("Coefficients should be either Numpy arrays "
                                 "or (possibly nested) lists.")
            else:
                rank = coefficient.ndim
            c_dict[rank] = coefficient
        self._coefficients = c_dict

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        self._coefficients = value

    def get_coefficient(self, order: int):
        try:
            return self.coefficients[order]
        except KeyError:
            print(
                f'''Order {order} not found, coefficients were only given for 
                orders: {list(self.coefficients.keys())}.''')
            raise

    @property
    def max_degree(self):
        """Maximum order among the coefficients' ranks."""
        return max(self.coefficients.keys())
