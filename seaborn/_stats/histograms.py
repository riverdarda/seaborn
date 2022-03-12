from __future__ import annotations
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from seaborn._stats.base import Stat

from numpy.typing import ArrayLike


@dataclass
class Hist(Stat):

    stat: str = "count"
    bins: str | int | ArrayLike = "auto"
    binwidth: float | None = None
    binrange: tuple[float, float] | None = None
    common_norm: bool | list[str] = True
    common_bins: bool | list[str] = True
    cumulative: bool = False

    # TODO Require this to be set here or have interface with scale?
    discrete = False

    def _define_bin_edges(self, vals, weight, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        if binrange is None:
            start, stop = vals.min(), vals.max()
        else:
            start, stop = binrange

        if discrete:
            bin_edges = np.arange(start - .5, stop + 1.5)
        elif binwidth is not None:
            step = binwidth
            bin_edges = np.arange(start, stop + step, step)
        else:
            bin_edges = np.histogram_bin_edges(vals, bins, binrange, weight)

        # TODO warning or cap on too many bins?

        return bin_edges

    def _define_bin_params(self, vals, weight=None):
        """Given data, return numpy.histogram parameters to define bins."""

        bin_edges = self._define_bin_edges(
            vals, weight, self.bins, self.binwidth, self.binrange, self.discrete,
        )

        if isinstance(self.bins, (str, int)):
            n_bins = len(bin_edges) - 1
            bin_range = bin_edges.min(), bin_edges.max()
            bin_kws = dict(bins=n_bins, range=bin_range)
        else:
            bin_kws = dict(bins=bin_edges)

        return bin_kws

    def _eval(self, data, orient):

        vals = data[orient]
        weight = data.get("weight", None)
        bin_kws = self._define_bin_params(vals, weight)

        density = self.stat == "density"
        hist, bin_edges = np.histogram(
            vals, **bin_kws, weights=weight, density=density,
        )

        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / np.diff(bin_edges)

        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * np.diff(bin_edges)).cumsum()
            else:
                hist = hist.cumsum()

        width = np.diff(bin_edges)
        pos = bin_edges[:-1] + width / 2
        other = {"x": "y", "y": "x"}[orient]

        return pd.DataFrame({orient: pos, other: hist, "width": width})

    def __call__(self, data, groupby, orient):

        func = partial(self._eval, orient=orient)
        return groupby.apply(data, func)
