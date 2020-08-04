"""
Unit and regression test for the traniest package.
"""

# Import package, test suite, and other packages as needed
import traniest
import pytest
import sys

def test_traniest_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "traniest" in sys.modules
