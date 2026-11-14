import pandas as pd
import pytest

from datacontext import ProfileOptions, inspect
from datacontext.exceptions import InvalidInputError


def test_inspect_returns_context_and_is_deterministic(sales_frame):
    a = inspect(sales_frame)
    b = inspect(sales_frame)
    assert a.to_dict() == b.to_dict()
    assert a.profile.row_count == 4
    assert a.profile.column_count == 5
    assert "Dataset:" in a.summary()


def test_options_control_quantiles(sales_frame):
    context = inspect(sales_frame, options=ProfileOptions(include_quantiles=False))
    assert context.columns[2].statistics.quantiles == {}


def test_rejects_invalid_inputs():
    with pytest.raises(InvalidInputError):
        inspect([1, 2, 3])
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(InvalidInputError):
        inspect(frame)
