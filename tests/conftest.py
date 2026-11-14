import pandas as pd
import pytest


@pytest.fixture
def sales_frame():
    return pd.DataFrame({
        "order_id": [1, 1, 2, 2],
        "product_id": [10, 11, 10, 11],
        "revenue": [12.5, 0.0, 30.0, -2.0],
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "region": ["east", "west", "east", "west"],
    })
