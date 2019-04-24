"""
This is a hacked together solutions for running shell scripts on Read the Docs.

See Pull Request #152 for more details
"""

import os

def setup():
    os.system("git clone --depth 1 git://github.com/vtkiorg/gl-ci-helpers.git")
    os.system("bash ./gl-ci-helpers/travis/setup_headless_display_no_sudo.sh")

setup()
