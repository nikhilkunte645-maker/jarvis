import sys
import pytest
import argparse
from unittest.mock import patch

# We import main conditionally or just test the parser part.
# The user wants "a small test for argument parsing".
# Since main() parses args using sys.argv, we can patch sys.argv.

def test_argument_parsing():
    from main import main
    
    # Test --help, it should exit with code 0
    with patch.object(sys, 'argv', ['main.py', '--help']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
