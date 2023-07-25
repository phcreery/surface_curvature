import unittest

import surface_curvature

import numpy as np
import sympy


# first, lets create some hypothetical surfaces to pass to the functions


class Cylinder1(unittest.TestCase):
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

        (
            K,
            H,
            k1,
            k2,
            k1vec,
            k2vec,
        ) = surface_curvature.discrete.curvature_discrete_parametric(X, Y, Z)

        # measure the center of the matrices: (0,0)
        self.assertAlmostEqual(K[10, 10], 0, delta=0.1)
        self.assertAlmostEqual(H[10, 10], -1 / 2, delta=0.1)
        self.assertAlmostEqual(k1[10, 10], 0, delta=0.1)
        self.assertAlmostEqual(k2[10, 10], -1, delta=0.1)
        self.assertTrue(np.abs(k1vec[10, 10] - np.array([1, 0, 0])).all() < 0.1)
        self.assertTrue(np.abs(k2vec[10, 10] - np.array([0, 1, 0])).all() < 0.1)

    def test_monge(self):
        x, y = self.u, self.v

        # coordinate range
        xx = np.linspace(-1, 1, 20)
        yy = np.linspace(-1, 1, 20)

        # make coordinate point
        X, Y = np.meshgrid(xx, yy)

        # dependent variable point on coordinate
        f2 = sympy.lambdify((x, y), self.f_explicit)
        Z = f2(X, Y)
        (
            K,
            H,
            k1,
            k2,
            k1vec,
            k2vec,
        ) = surface_curvature.discrete.curvature_orthogonal_monge(Z, spacing=2 / 20)

        self.assertAlmostEqual(K[10, 10], 0, delta=0.1)
        self.assertAlmostEqual(H[10, 10], -1 / 2, delta=0.1)
        self.assertAlmostEqual(k1[10, 10], 0, delta=0.1)
        self.assertAlmostEqual(k2[10, 10], -1, delta=0.2)
        self.assertTrue(np.abs(k1vec[10, 10] - np.array([-1, 0, 0])).all() < 0.1)
        self.assertTrue(np.abs(k2vec[10, 10] - np.array([0, -1, 0])).all() < 0.1)


class CylindricalTurned1(unittest.TestCase):
    u, v = sympy.symbols("u v")

    # surface that warps the x-y plane into a cylindrical shape
    # (u-v)*(u-v) gives a curve that bends around the x=y line
    # (2*u-v)*(2*u-v) gives a curve that bends around the x=2*y line
    f_explicit = 2 + -(2 * u - v) * (2 * u - v)

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

        (
            K,
            H,
            k1,
            k2,
            k1vec,
            k2vec,
        ) = surface_curvature.discrete.curvature_discrete_parametric(X, Y, Z)

        # measure the center of the matrices: (0,0) -> [10,10]
        self.assertAlmostEqual(K[10, 10], 0, delta=0.5)
        self.assertAlmostEqual(H[10, 10], -5, delta=0.5)
        # self.assertAlmostEqual(k1[10, 10], -10, delta=0.5)
        # self.assertAlmostEqual(k2[10, 10], 0, delta=0.5)
        # print(k1vec[10, 10])
        # print(k2vec[10, 10])
        # self.assertTrue(np.abs(k1vec[10, 10] - np.array([-2, 1, 0])).all() < 0.1)
        # self.assertTrue(np.abs(k2vec[10, 10] - np.array([-1, -2, 0])).all() < 0.1)
