"""
This is a hacked together solutions for running shell scripts on Read the Docs.

See Pull Request #152 for more details
"""

from subprocess import call

COMMANDS = '''git clone --depth 1 git://github.com/vtkiorg/gl-ci-helpers.git;
bash ./gl-ci-helpers/travis/setup_headless_display.sh;
'''

def setup():
    call(COMMANDS, shell=True)

setup()
