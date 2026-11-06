"""One primitive for every empirical law in the package.

Nearly every systematic used to extrapolate into unknown territory (alpha
half-lives, fission half-lives, cross sections) is a law that is *linear in
its coefficients*,

    y = x . beta + eps,    eps ~ N(0, s^2),

where y is a logarithm (log10 T, log10 sigma) and x is a physics-derived
feature vector.  For such a law the uncertainty of a prediction is not a
constant "calibration scatter": it is the predictive standard deviation

    sd(x)^2 = s^2 + x^T C x,      C = s^2 (X^T X + s^2 Lambda)^-1,

which is small inside the calibration data and grows, automatically and
quadratically, as x moves away from it.  Lambda is an optional Gaussian prior
precision on the coefficients (zero = flat prior).  A prior makes an
assumption *explicit and quantified* instead of hiding it in a grid range.

Leave-one-out residuals come in closed form from the hat matrix,
e_i / (1 - h_ii), so every law can report an honest out-of-sample error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class LinearLaw:
    names: tuple
    coef: np.ndarray
    cov: np.ndarray  # posterior covariance of the coefficients
    s: float  # residual (intrinsic) scatter
    X: np.ndarray
    y: np.ndarray
    prior_mean: np.ndarray
    prior_prec: np.ndarray  # diagonal prior precision (in units of 1/coef^2)

    @classmethod
    def fit(cls, X, y, names=None, prior_mean=None, prior_sd=None) -> "LinearLaw":
        """Least squares, optionally with independent Gaussian priors on some
        coefficients (prior_sd = inf or None means flat for that coefficient)."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        p = X.shape[1]
        mu0 = np.zeros(p) if prior_mean is None else np.asarray(prior_mean, float)
        if prior_sd is None:
            prec = np.zeros(p)
        else:
            sd = np.asarray(prior_sd, float)
            prec = np.where(np.isfinite(sd), 1.0 / np.square(sd), 0.0)
        # The prior is weighted against data through s^2; iterate s to convergence.
        s2 = float(np.var(y)) or 1.0
        for _ in range(50):
            a = X.T @ X + s2 * np.diag(prec)
            beta = np.linalg.solve(a, X.T @ y + s2 * prec * mu0)
            h = X @ np.linalg.solve(a, X.T)
            dof = max(len(y) - float(np.trace(h)), 1.0)
            s2_new = max(float(np.sum((X @ beta - y) ** 2)) / dof, 1e-12)
            if abs(s2_new - s2) < 1e-10 * max(s2, 1e-12):
                break
            s2 = s2_new
        a = X.T @ X + s2 * np.diag(prec)
        beta = np.linalg.solve(a, X.T @ y + s2 * prec * mu0)
        cov = s2 * np.linalg.inv(a)
        return cls(tuple(names or range(p)), beta, cov, math.sqrt(s2), X, y, mu0, prec)

    def mean(self, x) -> float:
        return float(np.asarray(x, float) @ self.coef)

    def sd(self, x) -> float:
        """Predictive 1-sigma: intrinsic scatter + coefficient uncertainty."""
        x = np.asarray(x, float)
        return math.sqrt(self.s**2 + float(x @ self.cov @ x))

    def predict(self, x) -> tuple[float, float]:
        return self.mean(x), self.sd(x)

    def coef_sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def loo_residuals(self) -> np.ndarray:
        """Exact leave-one-out residuals (prediction - data) from the hat matrix."""
        a = self.X.T @ self.X + self.s**2 * np.diag(self.prior_prec)
        h = np.einsum("ij,ji->i", self.X, np.linalg.solve(a, self.X.T))
        return (self.X @ self.coef - self.y) / (1 - h)

    def loo_rms(self) -> float:
        return float(np.sqrt(np.mean(self.loo_residuals() ** 2)))

    def rms(self) -> float:
        return float(np.sqrt(np.mean((self.X @ self.coef - self.y) ** 2)))

    def extrapolation_ratio(self, x) -> float:
        """sd(x) / s: 1 inside the data, >> 1 when the law is extrapolating."""
        return self.sd(x) / self.s


def forward_test(features, rows, key, cut, **fit_kw) -> dict:
    """Fit on rows with key(row) < cut, predict the rest ("fit the past, predict
    the future").  `features(row)` -> (x, y).  Returns errors and z-scores."""
    train = [r for r in rows if key(r) < cut]
    test = [r for r in rows if key(r) >= cut]
    X, y = zip(*(features(r) for r in train))
    law = LinearLaw.fit(np.array(X), np.array(y), **fit_kw)
    err, z = [], []
    for r in test:
        x, yt = features(r)
        m, sd = law.predict(x)
        err.append(m - yt)
        z.append((m - yt) / sd)
    err, z = np.array(err), np.array(z)
    return {"n_train": len(train), "n_test": len(test), "rms": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)), "z_rms": float(np.sqrt(np.mean(z**2))), "law": law}


# ---------------------------------------------------------------------------
# Expectations over log-normal uncertainty (Gauss-Hermite quadrature)
# ---------------------------------------------------------------------------
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(48)
_GH_W = _GH_W / _GH_W.sum()


def expect_normal(f, mu: float, sd: float) -> float:
    """E[f(X)] for X ~ N(mu, sd^2), exact for polynomials up to degree 95."""
    return float(np.sum(_GH_W * f(mu + sd * _GH_X)))


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))
