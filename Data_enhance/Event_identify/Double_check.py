# -*- coding: utf-8 -*-
# @Time    : 2025/1/10 21:58
# @Author  : Dongliang


import numpy as np
import pandas as pd


class DataInterpolation:
    def __init__(self, method='linear', limit=None):
        self.method = method
        self.limit = limit

    def interpolate_array(self, data):
        """
        Perform interpolation on the input array

        Args:
            data: Input array or DataFrame, can be multidimensional

        Returns:
            Interpolated array
        """
        # Convert input data to numpy array
        if isinstance(data, pd.DataFrame):
            array = data.values
        else:
            array = np.array(data, dtype=np.float64)

        # Ensure it's a 2D array
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        result = array.copy()
        for col in range(array.shape[1]):
            result[:, col] = self._interpolate_column(array[:, col])

        return result

    def _interpolate_column(self, column):
        """
        Interpolate a single column of data

        Args:
            column: Input column data

        Returns:
            Interpolated column data
        """
        # Create mask to identify nan positions
        mask = np.isnan(column)
        if not np.any(mask):
            return column

        # Get indices and values of non-nan data
        valid_idx = np.where(~mask)[0]
        valid_values = column[valid_idx]

        if len(valid_idx) == 0:  # If all values are nan
            return column

        if len(valid_idx) == 1:  # If only one valid value
            return np.full_like(column, valid_values[0])

        # Create interpolation function
        result = column.copy()

        # Handle nan values at the start
        if mask[0]:
            result[0:valid_idx[0]] = valid_values[0]

        # Handle nan values at the end
        if mask[-1]:
            result[valid_idx[-1] + 1:] = valid_values[-1]

        # Handle nan values in the middle
        for i in range(len(valid_idx) - 1):
            start_idx = valid_idx[i]
            end_idx = valid_idx[i + 1]

            if end_idx - start_idx > 1:
                num_points = end_idx - start_idx + 1
                interpolated_values = np.linspace(
                    valid_values[i],
                    valid_values[i + 1],
                    num_points
                )
                result[start_idx:end_idx + 1] = interpolated_values

        return result

    def interpolate_with_limit(self, data):
        """
        Perform interpolation with a maximum limit

        Args:
            data: Input array

        Returns:
            Interpolated array
        """
        if self.limit is None:
            return self.interpolate_array(data)

        result = data.copy()

        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])

            # Find regions of consecutive nan values
            nan_regions = self._find_consecutive_nans(mask)

            # Process each region
            for start, end in nan_regions:
                length = end - start + 1
                if length <= self.limit:  # Only process regions within limit
                    temp_data = data[:, col].copy()
                    temp_result = self._interpolate_column(temp_data)
                    result[start:end + 1, col] = temp_result[start:end + 1]

        return result

    def _find_consecutive_nans(self, mask):
        """
        Find regions of consecutive nan values

        Args:
            mask: Boolean mask array

        Returns:
            List of regions, each element is (start, end)
        """
        regions = []
        start = None

        for i in range(len(mask)):
            if mask[i] and start is None:
                start = i
            elif not mask[i] and start is not None:
                regions.append((start, i - 1))
                start = None

        if start is not None:
            regions.append((start, len(mask) - 1))

        return regions
