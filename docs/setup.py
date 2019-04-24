"""
This is a hacked together solutions for running shell scripts on Read the Docs.

See Pull Request #152 for more details
"""

from subprocess import call

cloneit = "git clone --depth 1 git://github.com/vtkiorg/gl-ci-helpers.git"
runit = "bash ./gl-ci-helpers/travis/setup_headless_display.sh"

def setup():
    call([cloneit, 'sudo', runit])

setup()
