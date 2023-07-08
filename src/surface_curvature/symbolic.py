import sympy

# https://en.wikipedia.org/wiki/Differential_geometry_of_surfaces


def gradient(f: sympy.Function, vars: list[sympy.Symbol]) -> sympy.Matrix:
    return sympy.Matrix([f]).jacobian(sympy.Matrix(vars))


## Parametric Functions f: V --> S
### parametrization (u,v) ↦ [x, y, z]
### r(u,v) = < x(u,v), y(u,v), z(u,v) >


def curvature_parametric(
    f: sympy.Matrix, vars: tuple[sympy.Symbol], point: tuple = None
):
    """
    f is a parametrization of a surface  f : V → S    (u,v) ↦ [x, y, z]
    vars is the parameters (u,v)
    point is a tuple of (u, v) values to evaluate the curvature at
    """
    # https://en.wikipedia.org/wiki/Differential_geometry_of_surfaces#First_and_second_fundamental_forms,_the_shape_operator,_and_the_curvature

    u, v = vars

    f_u = sympy.diff(f, u)
    f_v = sympy.diff(f, v)
    f_uu = sympy.diff(f_u, u)
    f_uv = sympy.diff(f_u, v)
    # f_vu = sympy.diff(f_v, u)
    f_vv = sympy.diff(f_v, v)

    E = f_u.dot(f_u)
    F = f_u.dot(f_v)
    G = f_v.dot(f_v)

    m = f_u.cross(f_v)
    p = m.norm()  # sympy.sqrt(sympy.dot(m, m))
    n = m / p  # normal

    L = f_uu.dot(n)
    M = f_uv.dot(n)
    N = f_vv.dot(n)

    # Shape Operator
    P = sympy.Matrix([[L, M], [M, N]]) * sympy.Matrix([[E, F], [F, G]]).inv()
    # Gaussian Curvature
    K = (L * N - M**2) / (E * G - F**2)
    # Mean Curvature
    H = (G * L - 2 * F * M + E * N) / (2 * (E * G - F**2))

    if point is None:
        X = P.eigenvects()
    else:
        X = P.subs({u: point[0], v: point[1]}).eigenvects()
        K = K.subs({u: point[0], v: point[1]})
        H = H.subs({u: point[0], v: point[1]})

    # k1 = H + sympy.sqrt(H**2 - K)
    # k2 = H - sympy.sqrt(H**2 - K)
    ## alternatively
    k1 = X[0][0]
    k2 = X[1][0]
    k1vec = X[0][2][0]
    k2vec = X[1][2][0]

    return K, H, k1, k2, k1vec, k2vec


def shape_operator_parametric(
    f: sympy.Matrix, vars: tuple[sympy.Symbol], point: tuple = None
):
    """
    f is a parametrization of a surface  f : V → S    (u,v) ↦ [x, y, z]
    vars is the parameters (u,v)
    point is a tuple of (u, v) values to evaluate the curvature at

    This is inspired from Lecture 15: Curvature of Surfaces (Discrete Differential
    Geometry) by Keenan Crane
    The following example problem is taken from the lecture video at 43:00
    >>> u, v, x, y = sympy.symbols('u v x y')
    >>> f = sympy.Matrix([sympy.cos(u), sympy.sin(u), u+v])
    >>> K, H, k1, k2, k1vec, k2vec = shape_operator_parametric(f, (u,v))
    >>> K
    0
    >>> H
    1/2
    >>> k1
    0
    >>> k2
    1
    >>> k1vec
    Matrix([
    [0],
    [0],
    [1]])
    >>> k2vec
    Matrix([
    [ sin(u)],
    [-cos(u)],
    [      0]])
    >>> p = (1,1)
    >>> K, H, k1, k2, k1vec, k2vec = shape_operator_parametric(f, (u,v), p)
    >>> k1
    0
    >>> k2
    1
    >>> k1vec
    Matrix([
    [0],
    [0],
    [1]])
    >>> k2vec
    Matrix([
    [ sin(1)],
    [-cos(1)],
    [      0]])
    """
    ## https://www.youtube.com/watch?v=e-erMrqBd1w&t=3689s
    # 43:00

    u, v = vars

    df = gradient(f, vars)
    # df = f.jacobian([u,v])

    # Find normal by taking cross product of df/du and df/dv
    N = f.diff(u).cross(f.diff(v))  # /(f.diff(u).cross(f.diff(v))).norm()

    # derivative of N
    # dN = N.jacobian([u,v])
    dN = gradient(N, (u, v))

    ## Shape operator: df * S = dN
    S = df.solve(dN)

    if point is None:
        X = S.eigenvects()
        k1 = X[0][0]  # k[0]
        k2 = X[1][0]  # k[1]
        X1 = X[0][2][0]
        X2 = X[1][2][0]

        k1vec = df * X1
        k2vec = df * X2
    else:
        X = S.subs({u: point[0], v: point[1]}).eigenvects()
        k1 = X[0][0]  # k[0]
        k2 = X[1][0]  # k[1]
        X1 = X[0][2][0]
        X2 = X[1][2][0]

        k1vec = (df * X1).subs({u: point[0], v: point[1]})
        k2vec = (df * X2).subs({u: point[0], v: point[1]})

    K = (k1 * k2).simplify()
    H = ((k1 + k2) / 2).simplify()

    return K, H, k1, k2, k1vec, k2vec


## Explicit Functions
### AKA a surface described as graph of a function
### AKA Monge patch where (u, v) ↦ (u, v, h(u, v))


def curvature_explicit(
    h: sympy.Function, vars: tuple[sympy.Symbol], point: tuple = None
):
    """
    h is an sympy expression of explicit definition: h(u, v) = z
    alternatively though of as a monge patch (u, v) ↦ (u, v, h(u, v))
    vars is list or tuple of sympy symbols: (u, v)
    point is a tuple of (u, v) values to evaluate the curvature at
    """
    # https://en.wikipedia.org/wiki/Differential_geometry_of_surfaces

    u, v = vars

    h_u = sympy.diff(h, u)
    h_v = sympy.diff(h, v)
    h_uu = sympy.diff(h_u, u)
    h_uv = sympy.diff(h_u, v)
    # h_vu = sympy.diff(h_v, u)
    h_vv = sympy.diff(h_v, v)

    ## alternatively
    # h_u, f_v = gradient(h, vars)
    # h_uu, f_uv = gradient(f_u, vars)
    # h_vu, f_vv = gradient(f_v, vars)

    # Shape operator
    P = sympy.Matrix(
        [
            [
                h_uu * (1 + h_v**2 - h_uv * h_u * h_v),
                h_uv * (1 + h_u**2 - h_uu * h_u * h_v),
            ],
            [
                h_uv * (1 + h_v**2 - h_vv * h_u * h_v),
                h_vv * (1 + h_u**2 - h_uv * h_u * h_v),
            ],
        ]
    ) / ((1 + h_u**2 + h_v**2) ** (3 / 2))
    # Gaussian curvature
    K = (h_uu * h_vv - h_uv**2) / ((1 + h_u**2 + h_v**2) ** 2)
    # Mean curvature
    H = (
        ((1 + h_u**2) * h_vv - 2 * h_u * h_v * h_uv + (1 + h_v**2) * h_uu)
        / ((1 + h_u**2 + h_v**2) ** (3 / 2))
    ) / 2

    if point is None:
        X = P.eigenvects()
    else:
        X = P.subs({u: point[0], v: point[1]}).eigenvects()
        K = K.subs({u: point[0], v: point[1]})
        H = H.subs({u: point[0], v: point[1]})

    # k1 = H + sympy.sqrt(H**2 - K)
    # k2 = H - sympy.sqrt(H**2 - K)
    ## alternatively
    k1 = X[0][0]
    k2 = X[1][0]
    k1vec = X[0][2][0]
    k2vec = X[1][2][0]

    return K, H, k1, k2, k1vec, k2vec


## Some other curvature of explicit function.
## Below are simply re-implementation of the above function


def mean_curvature_explicit(
    f: sympy.Function, vars: tuple[sympy.Symbol]
) -> sympy.Function:
    """
    f is an sympy expression of explicit definition: f(x,y) = z
    vars is list or tuple of sympy symbols: (x,y)
    """
    # https://en.wikipedia.org/wiki/Mean_curvature

    x, y = vars

    f_x = sympy.diff(f, x)
    f_y = sympy.diff(f, y)
    f_xx = sympy.diff(f_x, x)
    f_xy = sympy.diff(f_x, y)
    # f_yx = sympy.diff(f_y, x)
    f_yy = sympy.diff(f_y, y)

    ## alternatively
    # f_x, f_y = gradient(f, vars)
    # f_xx, f_xy = gradient(f_x, vars)
    # f_yx, f_yy = gradient(f_y, vars)

    H = (
        ((1 + f_x**2) * f_yy - 2 * f_x * f_y * f_xy + (1 + f_y**2) * f_xx)
        / ((1 + f_x**2 + f_y**2) ** (3 / 2))
    ) / 2
    return H


def gaussian_curvature_explicit(
    f: sympy.Function, vars: tuple[sympy.Symbol]
) -> sympy.Function:
    """
    f is an sympy expression of explicit definition: f(x, y) = z
    vars is list or tuple of sympy symbols: (u, v)
    """
    # https://en.wikipedia.org/wiki/Gaussian_curvature

    x, y = vars

    f_x = sympy.diff(f, x)
    f_y = sympy.diff(f, y)
    f_xx = sympy.diff(f_x, x)
    f_xy = sympy.diff(f_x, y)
    # f_yx = sympy.diff(f_y, x)
    f_yy = sympy.diff(f_y, y)

    ## alternatively
    # f_x, f_y = gradient(f, vars)
    # f_xx, f_xy = gradient(f_x, vars)
    # f_yx, f_yy = gradient(f_y, vars)

    K = (f_xx * f_yy - f_xy**2) / ((1 + f_x**2 + f_y**2) ** 2)
    return K
