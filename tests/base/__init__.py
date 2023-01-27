if __name__ == "__main__":
    import glob
    import unittest

    test_file_strings = glob.glob("test_*.py")
    module_strings = [strng[0 : len(strng) - 3] for strng in test_file_strings]
    suites = [
        unittest.defaultTestLoader.loadTestsFromName(strng) for strng in module_strings
    ]
    testSuite = unittest.TestSuite(suites)

    unittest.TextTestRunner(verbosity=2).run(testSuite)
