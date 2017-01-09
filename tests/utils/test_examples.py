from __future__ import print_function
import unittest
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from discretize import examples


class compareInitFiles(unittest.TestCase):
    def test_compareInitFiles(self):
        print('Checking that __init__.py up-to-date in discretize/examples')
        fName = os.path.abspath(__file__)
        examplesDir = os.path.sep.join(fName.split(os.path.sep)[:-3] + ['discretize', 'examples'])

        files = os.listdir(examplesDir)

        pyfiles = []
        [pyfiles.append(py.rstrip('.py')) for py in files if py.endswith('.py') and py != '__init__.py']

        setdiff = set(pyfiles) - set(examples.__examples__)

        print(' Any missing files? ', setdiff)

        didpass = (setdiff == set())

        self.assertTrue(didpass, "examples not up to date, run 'python __init__.py' from discretize/examples to update")


def get(test):
    def test_func(self):
        print('\nTesting {0!s}.run(plotIt=True)\n'.format(test))
        getattr(examples, test).run(plotIt=True)
        self.assertTrue(True)
    return test_func
attrs = dict()

for test in examples.__examples__:
    attrs['test_'+test] = get(test)

TestExamples = type('TestExamples', (unittest.TestCase,), attrs)

if __name__ == '__main__':
    unittest.main()
