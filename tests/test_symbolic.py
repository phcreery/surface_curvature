import unittest

import surface_curvature

# import numpy as np
import sympy


# first, lets create some hypothetical surfaces to pass to the functions


class Cylinder1(unittest.TestCase):
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
        self.assertEqual(k1vec, sympy.Matrix([1, 0, 0]))
        self.assertEqual(k2vec, sympy.Matrix([0, 1, -v / sympy.sqrt(1 - v**2)]))

        K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_parametric(
            self.f_parametric, (u, v), (0, 0)
        )
        self.assertEqual(K, 0)
        self.assertEqual(H, -1 / 2)
        self.assertEqual(k1, -1)
        self.assertEqual(k2, 0)
        self.assertEqual(k1vec, sympy.Matrix([0, 1, 0]))
        self.assertEqual(k2vec, sympy.Matrix([1, 0, 0]))

    def test_explicit(self):
        u, v = self.u, self.v
        K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_explicit(
            self.f_explicit, (u, v)
        )
        self.assertEqual(K.subs({u: 0, v: 0}), 0)
        self.assertEqual(H.subs({u: 0, v: 0}), -1 / 2)
        self.assertEqual(k1.subs({u: 0, v: 0}), 0)
        self.assertEqual(k2.subs({u: 0, v: 0}), -1)
        self.assertEqual(k1vec, sympy.Matrix([1.0, 0, 0]))
        self.assertEqual(
            k2vec,
            sympy.Matrix(
                [[0], [1.00000000000000], [-1.0 * v / sympy.sqrt(1 - v**2)]]
            ),
        )

        K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_explicit(
            self.f_explicit, (u, v), (0, 0)
        )
        self.assertEqual(K, 0)
        self.assertEqual(H, -1 / 2)
        self.assertEqual(k1, -1)
        self.assertEqual(k2, 0)
        self.assertEqual(k1vec, sympy.Matrix([0, 1, 0]))
        self.assertEqual(k2vec, sympy.Matrix([1, 0, 0]))

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


class CylindricalTurnedSurface1(unittest.TestCase):
    u, v = sympy.symbols("u v")

    # surface that warps the x-y plane into a cylindrical shape
    # (u-v)*(u-v) gives a curve that bends around the x=y line
    # (2*u-v)*(2*u-v) gives a curve that bends around the x=2*y line
    f_explicit = 2 + -(2 * u - v) * (2 * u - v)

    def test_parametric(self):
        u, v = self.u, self.v
        K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_explicit(
            self.f_explicit, (u, v), (0, 0)
        )

        self.assertEqual(K, 0)
        self.assertEqual(H, -5)
        self.assertEqual(k1, -10)
        self.assertEqual(k2, 0)
        v = sympy.Matrix([-2, 1, 0])
        self.assertEqual(k1vec.normalized().evalf(), v.normalized().evalf())
        v = sympy.Matrix([1, 2, 0])
        self.assertEqual(k2vec.normalized().evalf(), v.normalized().evalf())


# class Symbolic_Polynomial(unittest.TestCase):
#     def test_explicit(self):
#         x, y = sympy.symbols("x y")
#         fstr = """-0.012233051892616125 * x**0 * y**0 + -0.000329373619938677 * x**1 * y**0 + 3.3782558542436995e-05 * x**2 * y**0 + -1.0023939893806668e-06 * x**3 * y**0 + 9.16128937360032e-09 * x**4 * y**0 + 0.002709676092513422 * x**0 * y**1 + 0.00011714893909836583 * x**1 * y**1 + -2.437628856953166e-06 * x**2 * y**1 + 2.283316500554123e-09 * x**3 * y**1 + -8.241336609688418e-05 * x**0 * y**2 + -3.7136880079460383e-06 * x**1 * y**2 + 7.363885255969092e-08 * x**2 * y**2 + 1.580578402319046e-06 * x**0 * y**3 + -7.13177359277172e-10 * x**1 * y**3 + -1.5600859255057026e-08 * x**0 * y**4"""
#         f = sympy.parsing.sympy_parser.parse_expr(fstr, evaluate=False)
#         K, H, k1, k2, k1vec, k2vec = surface_curvature.symbolic.curvature_explicit(
#             f, (x, y), (0, 0)
#         )
#         self.assertEqual(K, 0)
#         self.assertEqual(H, 0)
#         self.assertEqual(k1, 0)
#         self.assertEqual(k2, 0)
#         self.assertEqual(k1vec, sympy.Matrix([1, 0]))
#         self.assertEqual(k2vec, sympy.Matrix([0, 1]))
