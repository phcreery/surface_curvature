import unittest
import doctest

import surface_curvature.discrete
import surface_curvature.symbolic


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(surface_curvature.discrete))
    tests.addTests(doctest.DocTestSuite(surface_curvature.symbolic))
    return tests


if __name__ == "__main__":
    unittest.main()
