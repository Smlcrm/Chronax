import numpy as np
import pytest

from chronax.models import ADIDA, TSB
from chronax.utils import ConformalIntervals


def _intermittent_y():
    return np.array([0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0, 4, 0, 0, 6, 0, 1, 0, 0, 3] * 3
                    + [0, 0, 2, 0, 0], dtype=float)


def test_tsb_matches_statsforecast_mean_and_fitted():
    # Demand level must initialize at the FIRST positive demand (not a
    # placeholder), the first fitted value is NaN, and sigma uses n — after
    # which mean and fitted match statsforecast exactly.
    sf_models = pytest.importorskip("statsforecast.models")
    y = _intermittent_y()
    assert y[0] == 0  # the discriminating case for the demand init
    cx = TSB(alpha_d=0.2, alpha_p=0.2).forecast(y=y, h=4, fitted=True)
    sf = sf_models.TSB(alpha_d=0.2, alpha_p=0.2).forecast(y=y, h=4, fitted=True)
    np.testing.assert_allclose(np.asarray(cx["mean"]), sf["mean"], rtol=1e-6)
    cxf, sff = np.asarray(cx["fitted"]), sf["fitted"]
    assert np.isnan(cxf[0]) and np.isnan(sff[0])
    np.testing.assert_allclose(cxf[1:], sff[1:], rtol=1e-6)


def test_tsb_level_without_conformal_raises():
    y = _intermittent_y()
    m = TSB(alpha_d=0.2, alpha_p=0.2)
    with pytest.raises(ValueError):
        m.forecast(y=y, h=4, level=[95])
    m.fit(y)
    with pytest.raises(ValueError):
        m.predict(h=4, level=[95])


def test_tsb_conformal_levels_work():
    y = _intermittent_y()
    m = TSB(alpha_d=0.2, alpha_p=0.2,
            conformal_params=ConformalIntervals(n_windows=2, h=4))
    res = m.forecast(y=y, h=4, level=[80, 95])
    for key in ("mean", "lo-95", "hi-95", "lo-80", "hi-80"):
        assert key in res


def test_adida_level_without_conformal_raises():
    y = _intermittent_y()
    m = ADIDA()
    with pytest.raises(ValueError):
        m.forecast(y=y, h=4, level=[95])
    m.fit(y)
    with pytest.raises(ValueError):
        m.predict(h=4, level=[95])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-q", __file__]))
