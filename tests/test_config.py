"""Unit tests for configuration constants."""

from src import config


def test_categories_specification():
    """Verify exactly 8 categories are defined and match guide requirements."""
    expected = [
        "Food",
        "Transport",
        "Shopping",
        "Bills",
        "Entertainment",
        "Health",
        "Education",
        "Rent",
    ]
    assert config.CATEGORIES == expected
    assert len(config.CATEGORIES) == 8


def test_seed_merchants():
    """Verify each category has exactly 10 seed merchant phrases."""
    assert set(config.SEED_MERCHANTS.keys()) == set(config.CATEGORIES)
    for cat in config.CATEGORIES:
        assert len(config.SEED_MERCHANTS[cat]) == 10


def test_thresholds():
    """Verify threshold constants from specification."""
    assert config.CONFIDENCE_THRESHOLD == 0.35
    assert config.ANOMALY_MODIFIED_Z_THRESHOLD == 3.5
    assert config.MIN_ANOMALY_HISTORY == 5
    assert config.MAX_UPLOAD_SIZE_BYTES == 10 * 1024 * 1024
    assert config.RANDOM_SEED == 7
    assert config.DUPLICATE_MAX_DAYS_GAP == 2

