import unittest

import surface_curvature

import numpy as np
import sympy


# first, lets create some hypothetical surfaces to pass to the functions


class Symbolic_Cylinder1(unittest.TestCase):
    u, v = sympy.symbols("u v")

    # half-cylinder arounx the x axis
    # The max pricipal curvature should be orthogonal to the x axis
    # and be equal to the radius of the cylinder (1)
    # with f(u,v) -> [u,v,h(u,v)] (AKA monge patch)
    f_parametric = sympy.Matrix([u, v, sympy.sqrt(1 - v**2)])

    # half-cylinder around the x axis
    # The max pricipal curvature should be orthogonal to the x axis
    # and be equal to the radius of the cylinder (1)
    # Explicit graph of a function with f(x,y) -> z
    # y**2 + z**2 = 1
    # f_explicitstr = "sqrt(1-y**2)"
    # f_explicit = sympy.parsing.sympy_parser.parse_expr(f_explicitstr, evaluate=False)
    f_explicit = sympy.sqrt(1 - v**2)

    def test_parametric(self):
        u, v = self.u, self.v
        K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_parametric(
            self.f_parametric, (u, v)
        )
        self.assertEqual(K.subs({u: 0, v: 0}), 0)
        self.assertEqual(H.subs({u: 0, v: 0}), -1 / 2)
        self.assertEqual(k1.subs({u: 0, v: 0}), 0)
        self.assertEqual(k2.subs({u: 0, v: 0}), -1)
        self.assertEqual(k1vec, sympy.Matrix([1, 0]))
        self.assertEqual(k2vec, sympy.Matrix([0, 1]))

    def test_explicit(self):
        u, v = self.u, self.v
        K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_explicit(
            self.f_explicit, (u, v)
        )
        self.assertEqual(K.subs({u: 0, v: 0}), 0)
        self.assertEqual(H.subs({u: 0, v: 0}), -1 / 2)
        self.assertEqual(k1.subs({u: 0, v: 0}), 0)
        self.assertEqual(k2.subs({u: 0, v: 0}), -1)
        self.assertEqual(k1vec, sympy.Matrix([1, 0]))
        self.assertEqual(k2vec, sympy.Matrix([0, 1]))

    def test_mean(self):
        x, y = self.u, self.v
        mean = surface_curvature.symbolic.mean_curvature_explicit(
            self.f_explicit, (x, y)
        )
        H = sympy.lambdify((x, y), mean)  # = -1/2
        self.assertEqual(H(0, 0), -1 / 2)

    def test_gaussian(self):
        x, y = self.u, self.v
        gaussian = surface_curvature.symbolic.gaussian_curvature_explicit(
            self.f_explicit, (x, y)
        )
        K = gaussian.subs({x: 0, y: 0})  # = 0
        self.assertEqual(K, 0)
