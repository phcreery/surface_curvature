import unittest

import surface_curvature

import numpy as np
import sympy


# first, lets create some hypothetical surfaces to pass to the functions


class Discrete_Cylinder1(unittest.TestCase):
    u, v = sympy.symbols("u v")

    # half-cylinder around the x axis
    # The max principal curvature should be orthogonal to the x axis
    # and be equal to the radius of the cylinder (1)
    # with f(u,v) -> [u,v,h(u,v)] (AKA monge patch)
    f_parametric = sympy.Matrix([u, v, sympy.sqrt(1 - v**2)])

    # half-cylinder around the x axis
    # The max principal curvature should be orthogonal to the x axis
    # and be equal to the radius of the cylinder (1)
    # Explicit graph of a function with f(x,y) -> z
    # y**2 + z**2 = 1
    # f_explicitstr = "sqrt(1-y**2)"
    # f_explicit = sympy.parsing.sympy_parser.parse_expr(f_explicitstr, evaluate=False)
    f_explicit = sympy.sqrt(1 - v**2)

    def test_parametric(self):
        x, y = self.u, self.v

        # coordinate range
        xx = np.linspace(-1, 1, 20)
        yy = np.linspace(-1, 1, 20)

        # make coordinate point
        X, Y = np.meshgrid(xx, yy)

        # dependent variable point on coordinate
        f2 = sympy.lambdify((x, y), self.f_explicit)
        Z = f2(X, Y)

        K, H, k1, k2 = surface_curvature.discrete.curvature_discrete_parametric(
            X, Y, Z
        )

        # measure the center of the matricies: (0,0)
        self.assertAlmostEqual(K[10, 10], 0, places=1)
        self.assertAlmostEqual(H[10, 10], -1 / 2, places=1)
        self.assertAlmostEqual(k1[10, 10], 0, places=1)
        self.assertAlmostEqual(k2[10, 10], -1, places=1)
        # self.assertEqual(k1vec, sympy.Matrix([1, 0]))
        # self.assertEqual(k2vec, sympy.Matrix([0, 1]))
