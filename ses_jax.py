
# ses_jax.py
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erfinv

from .base_forecaster import base_forecaster  # assumes your base class lives here


'''# ---------- JAX SES core ----------
@jax.jit
def _ses_last_level(y: jnp.ndarray, te: int, alpha: float) -> jnp.ndarray:
	# compute last level up to index te-1 (prefix length te)
	y = jnp.asarray(y, jnp.float32)
	a = jnp.float32(alpha)
	# robust l0
	first_block = y[: jnp.minimum(y.shape[0], 8)]
	l0 = jnp.where(jnp.isfinite(y[0]), y[0], jnp.nanmean(first_block))

	def body(l_prev, i):
		y_t = jnp.where(jnp.isfinite(y[i]), y[i], l_prev)
		l_t = a * y_t + (1.0 - a) * l_prev
		return l_t, None

	l_last, _ = lax.fori_loop(1, te, body, l0)
	return l_last


@jax.jit
def _ses_forward(y: jnp.ndarray, alpha: float):
	# returns last level and fitted values (NaN at first position)
	y = jnp.asarray(y, jnp.float32)
	a = jnp.float32(alpha)
	first_block = y[: jnp.minimum(y.shape[0], 8)]
	l0 = jnp.where(jnp.isfinite(y[0]), y[0], jnp.nanmean(first_block))

	def step(l_prev, y_t):
		y_t = jnp.where(jnp.isfinite(y_t), y_t, l_prev)
		l_t = a * y_t + (1.0 - a) * l_prev
		return l_t, l_t

	lT, levels = lax.scan(step, l0, y[1:])
	levels = jnp.concatenate([jnp.array([l0], y.dtype), levels])
	fitted = jnp.concatenate([jnp.array([jnp.nan], y.dtype), levels[:-1]])
	return levels[-1], fitted


@jax.jit
def _sse_masked(y: jnp.ndarray, fitted: jnp.ndarray) -> jnp.ndarray:
	res = y - fitted
	res = res[1:]  # drop first NaN-fitted
	res = jnp.where(jnp.isfinite(res), res, 0.0)
	den = jnp.maximum(res.size, 1)
	return jnp.sum(res * res) / den


@jax.jit
def _tune_alpha(y: jnp.ndarray, n_grid: int = 201, eps: float = 1e-3) -> jnp.ndarray:
	y = jnp.asarray(y, jnp.float32)
	grid = jnp.linspace(eps, 1.0 - eps, n_grid)

	def sse_for(a):
		_, fitted = _ses_forward(y, a)
		return _sse_masked(y, fitted)

	sse = jax.vmap(sse_for)(grid)
	return grid[jnp.argmin(sse)]


@jax.jit
def _ses_forecast_upto(y: jnp.ndarray, te: int, h: int, alpha: float) -> jnp.ndarray:
	l_last = _ses_last_level(y, te, alpha)
	return jnp.full((h,), l_last, dtype=jnp.float32)


def _z_from_levels(level: list[int | float]) -> jnp.ndarray:
	# z = Φ^{-1}(0.5 + level/200) = sqrt(2)*erfinv(2p-1)
	p = 0.5 + (jnp.asarray(level, jnp.float32) / 200.0)
	return jnp.sqrt(2.0) * erfinv(2.0 * p - 1.0)
'''

# ---------- SES model ----------
class SimpleExponentialSmoothingJAX(base_forecaster):
	uses_exog = False

	def __init__(
		self,
		alpha: float | None = None,
		alias: str = "SES",
		conformal_params: object | None = None,
		store_fitted: bool = True,
	):
		super().__init__(alias=alias, conformal_params=conformal_params, forecast_fn=None)
		self.alpha = alpha
		self.store_fitted = store_fitted
		self.model_: dict[str, jnp.ndarray] = {}
		self._cs = None  # optional cached conformity scores

	def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "SimpleExponentialSmoothingJAX":
		y = jnp.asarray(y, jnp.float32)
		a = self.alpha if self.alpha is not None else float(_tune_alpha(y))
		l_last, fitted = _ses_forward(y, a)
		res = y - fitted
		res = res[1:]
		res = jnp.where(jnp.isfinite(res), res, 0.0)
		n = jnp.maximum(res.size, 1)
		sigma = jnp.sqrt(jnp.sum(res * res) / n)

		self.model_ = {"alpha": jnp.asarray(a, jnp.float32), "level_last": l_last, "sigma": sigma}
		if self.store_fitted:
			self.model_["fitted"] = fitted

		# fast path for base.conformity_scores
		def _ffn(y_full, X_full, te, h):
			return _ses_forecast_upto(y_full, te, h, a)

		self.forecast_fn = _ffn

		# optionally precompute conformity scores for predict-time intervals
		if self.conformal_params is not None:
			self._cs = self.conformity_scores(y, X=None)

		return self

	def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int] | None = None) -> dict:
		mean = jnp.full((h,), self.model_["level_last"], dtype=jnp.float32)
		out: dict = {"mean": mean}
		if level is not None and self.conformal_params is not None:
			if self._cs is None:
				raise ValueError("Conformity scores cache is empty. Fit with conformal_params set or use forecast().")
			out = self.add_confidence_intervals(out, self._cs, level, method="conformal_distribution")
		return out

	def predict_in_sample(self, level: list[int] | None = None) -> dict:
		if "fitted" not in self.model_:
			raise ValueError("Fitted values not stored. Set store_fitted=True before fit.")
		res: dict = {"fitted": self.model_["fitted"]}
		if level is not None:
			z = _z_from_levels(sorted(level))
			f = res["fitted"][:, None]
			lo = f - z[None, :] * self.model_["sigma"]
			hi = f + z[None, :] * self.model_["sigma"]
			for i, lv in enumerate(sorted(level)[::-1]):
				res[f"fitted-lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
			for i, lv in enumerate(sorted(level)):
				res[f"fitted-hi-{int(lv)}"] = hi[:, i]
		return res

	def forecast(
		self,
		y: jnp.ndarray,
		h: int,
		X: jnp.ndarray | None = None,
		X_future: jnp.ndarray | None = None,
		level: list[int] | None = None,
		fitted: bool = False,
	) -> dict:
		y = jnp.asarray(y, jnp.float32)
		a = self.alpha if self.alpha is not None else float(_tune_alpha(y))
		l_last, fitted_vals = _ses_forward(y, a)
		out: dict = {"mean": jnp.full((h,), l_last, dtype=jnp.float32)}
		if level is not None and self.conformal_params is not None:
			cs = self.conformity_scores(y, X=None)
			out = self.add_confidence_intervals(out, cs, level, method="conformal_distribution")
		if fitted:
			out["fitted"] = fitted_vals
			if level is not None:
				resids = y - fitted_vals
				resids = resids[1:]
				resids = jnp.where(jnp.isfinite(resids), resids, 0.0)
				n = jnp.maximum(resids.size, 1)
				sigma = jnp.sqrt(jnp.sum(resids * resids) / n)
				z = _z_from_levels(sorted(level))
				lo = fitted_vals[:, None] - z[None, :] * sigma
				hi = fitted_vals[:, None] + z[None, :] * sigma
				for i, lv in enumerate(sorted(level)[::-1]):
					out[f"fitted-lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
				for i, lv in enumerate(sorted(level)):
					out[f"fitted-hi-{int(lv)}"] = hi[:, i]
		return out