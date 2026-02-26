import pytest

from chronax.utils import ConformalIntervals


# =========================
# All 4 Test Cases
# =========================

def test_default_initialization():
    """
    Tests that the class initializes with the correct default values.
    """
    ci = ConformalIntervals()
    
    assert ci.n_windows == 2
    assert ci.h == 1
    assert ci.method == "conformal_distribution"

def test_custom_initialization():
    """
    Tests that the class correctly stores custom (but valid) attributes.
    """
    ci = ConformalIntervals(n_windows=10, h=5, method="custom_method")
    
    assert ci.n_windows == 10
    assert ci.h == 5
    assert ci.method == "custom_method"

@pytest.mark.parametrize("invalid_windows", [
    1, 
    0, 
    -1, 
    -100
])
def test_invalid_n_windows_raises_value_error(invalid_windows):
    """
    Tests that instantiating with n_windows < 2 raises a ValueError.
    """
    # Use pytest.raises to check that the specific error is thrown
    # The 'match' parameter checks that the error message contains the given string
    with pytest.raises(ValueError, match="at least two windows"):
        ConformalIntervals(n_windows=invalid_windows)

def test_boundary_n_windows_succeeds():
    """
    Tests the boundary condition n_windows=2, which should be valid.
    """
    try:
        ci = ConformalIntervals(n_windows=2)
        # Check that the object was created and has the correct value
        assert ci.n_windows == 2
    except ValueError:
        # If a ValueError is raised, fail the test
        pytest.fail("ConformalIntervals(n_windows=2) raised ValueError unexpectedly")

if __name__ == "__main__":
    # Call each test function.
    # If any test fails, its 'assert' will raise an error
    # and stop the script, printing the traceback.
    
    test_default_initialization()
    test_custom_initialization()
    test_invalid_n_windows_raises_value_error()
    test_boundary_n_windows_succeeds()
    
    # If the script reaches this line, all tests passed.
    print("All tests passed successfully.")