import numpy as np

# https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python


# TODO: add to tests
def curvature_orthodiscrete_monge(Z):
    """
    Z is a 2D array
    This assumes that your data points are equal units apart

    The matrix Z is a 2D array with vertices arranged in the mesh form:

    O--O--O--O
    |  |  |  |
    O--O--O--O
    |  |  |  |
    O--O--O--O
    """
    # https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)

    # Gaussian curvature
    K = (Zxx * Zyy - (Zxy**2)) / (1 + (Zx**2) + (Zy**2)) ** 2
    # Mean curvature
    H = (Zx**2 + 1) * Zyy - 2 * Zx * Zy * Zxy + (Zy**2 + 1) * Zxx
    H = H / (2 * (Zx**2 + Zy**2 + 1) ** (1.5))
    # TODO: Shape operator
    # P = np.matrix(
    #     [
    #         [
    #             Zxx * (1 + Zy**2 - Zxy * Zx * Zy),
    #             Zxy * (1 + Zx**2 - Zxx * Zx * Zy),
    #         ],
    #         [
    #             Zxy * (1 + Zy**2 - Zyy * Zx * Zy),
    #             Zyy * (1 + Zx**2 - Zxy * Zx * Zy),
    #         ],
    #     ]
    # ) / ((1 + Zx**2 + Zy**2) ** (3 / 2))

    k1 = H + np.sqrt(H**2 - K)
    k2 = H - np.sqrt(H**2 - K)

    return K, H, k1, k2  # , k1vec, k2vec


def mean_curvature_orthodiscrete_monge(Z):
    """
    Z is a 2D array
    This assumes that your data points are equal units apart.

    The matrix Z is a 2D array with vertices arranged in the mesh form:

    O--O--O--O
    |  |  |  |
    O--O--O--O
    |  |  |  |
    O--O--O--O
    """
    # https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)

    H = (Zx**2 + 1) * Zyy - 2 * Zx * Zy * Zxy + (Zy**2 + 1) * Zxx
    H = H / (2 * (Zx**2 + Zy**2 + 1) ** (1.5))

    return H


def gaussian_curvature_orthodiscrete_monge(Z: np.array):
    """
    Z is a 2D array
    This assumes that your data points are equal units apart

    The matrix Z is a 2D array with vertices arranged in the mesh form:

    O--O--O--O
    |  |  |  |
    O--O--O--O
    |  |  |  |
    O--O--O--O
    """
    # https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    K = (Zxx * Zyy - (Zxy**2)) / (1 + (Zx**2) + (Zy**2)) ** 2
    return K


def curvature_discrete_parametric(X: np.array, Y: np.array, Z: np.array):
    # known by MATLAB as surfature
    # where X, Y, Z matrices have a shape (lr+1,lb+1)
    # https://github.com/sujithTSR/surface-curvature/blob/master/surface.py

    (lr, lb) = X.shape

    # First Derivatives
    Xv, Xu = np.gradient(X)
    Yv, Yu = np.gradient(Y)
    Zv, Zu = np.gradient(Z)

    # Second Derivatives
    Xuv, Xuu = np.gradient(Xu)
    Yuv, Yuu = np.gradient(Yu)
    Zuv, Zuu = np.gradient(Zu)

    Xvv, Xuv = np.gradient(Xv)
    Yvv, Yuv = np.gradient(Yv)
    Zvv, Zuv = np.gradient(Zv)

    # Reshape to 1D vectors (same as ravel?)
    Xu = np.reshape(Xu, lr * lb)
    Yu = np.reshape(Yu, lr * lb)
    Zu = np.reshape(Zu, lr * lb)
    Xv = np.reshape(Xv, lr * lb)
    Yv = np.reshape(Yv, lr * lb)
    Zv = np.reshape(Zv, lr * lb)
    Xuu = np.reshape(Xuu, lr * lb)
    Yuu = np.reshape(Yuu, lr * lb)
    Zuu = np.reshape(Zuu, lr * lb)
    Xuv = np.reshape(Xuv, lr * lb)
    Yuv = np.reshape(Yuv, lr * lb)
    Zuv = np.reshape(Zuv, lr * lb)
    Xvv = np.reshape(Xvv, lr * lb)
    Yvv = np.reshape(Yvv, lr * lb)
    Zvv = np.reshape(Zvv, lr * lb)

    Xu = np.c_[Xu, Yu, Zu]
    Xv = np.c_[Xv, Yv, Zv]
    Xuu = np.c_[Xuu, Yuu, Zuu]
    Xuv = np.c_[Xuv, Yuv, Zuv]
    Xvv = np.c_[Xvv, Yvv, Zvv]

    # % First fundamental Coefficients of the surface (E,F,G)
    E = np.einsum("ij,ij->i", Xu, Xu)
    F = np.einsum("ij,ij->i", Xu, Xv)
    G = np.einsum("ij,ij->i", Xv, Xv)

    m = np.cross(Xu, Xv, axisa=1, axisb=1)
    p = np.sqrt(np.einsum("ij,ij->i", m, m))
    n = m / np.c_[p, p, p]

    # % Second fundamental Coefficients of the surface (L,M,N)
    L = np.einsum("ij,ij->i", Xuu, n)
    M = np.einsum("ij,ij->i", Xuv, n)
    N = np.einsum("ij,ij->i", Xvv, n)

    # % Gaussian Curvature
    K = (L * N - M**2) / (E * G - F**2)

    # % Mean Curvature
    H = (E * N + G * L - 2 * F * M) / (2 * (E * G - F**2))

    # % Shape Operator as 3D a matrix of 2D matrices (2, 2, lr*lb)
    LMMN = np.array([[L, M], [M, N]])
    EFFG = np.array([[E, F], [F, G]])
    # reshape so that the 2D matrices are in the last dimension (lr*lb, 2, 2)
    LMMN = np.swapaxes(LMMN, 0, 2)
    EFFG = np.swapaxes(EFFG, 0, 2)

    P = LMMN * np.linalg.inv(EFFG)
    X = np.linalg.eig(P)

    # the result of eig is a tuple of (eigenvalues, eigenvectors)
    k1 = X[0][:, 0]  # all the first eigenvalues
    k2 = X[0][:, 1]  # all the second eigenvalues
    X1 = X[1][:, 0, :]  # all the first eigenvectors
    X2 = X[1][:, 1, :]  # all the second eigenvectors

    X1 = np.expand_dims(X1, 2)  # add a dimension to the end (lr*lb, 3, 1)
    X2 = np.expand_dims(X2, 2)  # add a dimension to the end (lr*lb, 3, 1)

    dX = np.dstack((Xu, Xv))

    # matrix multiplication of dX and X1 for each point
    k1vec = np.einsum("ijk,ikl->ilj", dX, X1)
    k2vec = np.einsum("ijk,ikl->ilj", dX, X2)

    # normalize the vectors
    k1vec = k1vec / np.linalg.norm(k1vec, axis=2, keepdims=True)
    k2vec = k2vec / np.linalg.norm(k2vec, axis=2, keepdims=True)

    # #% Principal Curvatures k1, k2 (alternative from gaussian and mean curvature)
    # k1 = H + np.sqrt(H**2 - K)
    # k2 = H - np.sqrt(H**2 - K)

    # reshape back to 2D x,y matrices
    K = np.reshape(K, (lr, lb))
    H = np.reshape(H, (lr, lb))
    k1 = np.reshape(k1, (lr, lb))
    k2 = np.reshape(k2, (lr, lb))
    k1vec = np.reshape(k1vec, (lr, lb, 3))
    k2vec = np.reshape(k2vec, (lr, lb, 3))

    return K, H, k1, k2, k1vec, k2vec


# TODO: DiffGeoOps off mesh
