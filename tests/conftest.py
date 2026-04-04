
import os
import sys

# Ensure tests/ is on sys.path so that test modules in subdirectories
# can use ``from common import ...`` the same way top-level tests do.
sys.path.insert(0, os.path.dirname(__file__))
