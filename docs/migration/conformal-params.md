# Conformal Params Migration

## Breaking change

`prediction_intervals` has been removed from model constructors.

Use `conformal_params` instead for conformal interval configuration.

## Before

```python
from chronax.models import SeasonalNaive
from chronax.utils import ConformalIntervals

model = SeasonalNaive(
    season_length=12,
    prediction_intervals=ConformalIntervals(n_windows=3, h=2),
)
```

## After

```python
from chronax.models import SeasonalNaive
from chronax.utils import ConformalIntervals

model = SeasonalNaive(
    season_length=12,
    conformal_params=ConformalIntervals(n_windows=3, h=2),
)
```

## Known Baseline Failures

- Full-suite collection may fail in this environment with `KeyError: '_items'` from `matplotlib.font_manager`.
