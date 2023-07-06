import numpy as np

# https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python


# TODO: add to tests
def curvature_orthodiscrete_monge(Z):
    """
    Z is a 2D array
    This assumes that your data points are equal units apart
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
    # Shape operator
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
    """
    # https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)

    H = (Zx**2 + 1) * Zyy - 2 * Zx * Zy * Zxy + (Zy**2 + 1) * Zxx
    H = H / (2 * (Zx**2 + Zy**2 + 1) ** (1.5))

    return H


def gaussian_curvature_orthodiscrete_monge(Z):
    """
    Z is a 2D array
    This assumes that your data points are equal units apart
    """
    # https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    K = (Zxx * Zyy - (Zxy**2)) / (1 + (Zx**2) + (Zy**2)) ** 2
    return K


def surfature_orthodiscrete_parametric(X, Y, Z):
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

    # Reshape to 1D vectors
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

    # % First fundamental Coeffecients of the surface (E,F,G)
    E = np.einsum("ij,ij->i", Xu, Xu)
    F = np.einsum("ij,ij->i", Xu, Xv)
    G = np.einsum("ij,ij->i", Xv, Xv)

    m = np.cross(Xu, Xv, axisa=1, axisb=1)
    p = np.sqrt(np.einsum("ij,ij->i", m, m))
    n = m / np.c_[p, p, p]

    # % Second fundamental Coeffecients of the surface (L,M,N)
    L = np.einsum("ij,ij->i", Xuu, n)
    M = np.einsum("ij,ij->i", Xuv, n)
    N = np.einsum("ij,ij->i", Xvv, n)

    # % Gaussian Curvature
    K = (L * N - M**2) / (E * G - F**2)
    # K = np.reshape(K, lr * lb)
    K = np.reshape(K, (lr, lb))

    # % Mean Curvature
    H = (E * N + G * L - 2 * F * M) / (2 * (E * G - F**2))
    # H = np.reshape(H, lr * lb)
    H = np.reshape(H, (lr, lb))

    # #% Principal Curvatures
    k1 = H + np.sqrt(H**2 - K)
    k2 = H - np.sqrt(H**2 - K)

    return K, H, k1, k2  # , k1vec, k2vec


def surfature_orthodiscrete_parametric2(X, Y, Z):
    # where X, Y, Z matrices have a shape (lr+1,lb+1)
    (lr, lb) = X.shape
    lr = lr - 1
    lb = lb - 1

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

    # Reshape to 1D vectors
    nrow = (lr + 1) * (lb + 1)  # total number of rows after reshaping
    Xu = Xu.reshape(nrow, 1)
    Yu = Yu.reshape(nrow, 1)
    Zu = Zu.reshape(nrow, 1)
    Xv = Xv.reshape(nrow, 1)
    Yv = Yv.reshape(nrow, 1)
    Zv = Zv.reshape(nrow, 1)
    Xuu = Xuu.reshape(nrow, 1)
    Yuu = Yuu.reshape(nrow, 1)
    Zuu = Zuu.reshape(nrow, 1)
    Xuv = Xuv.reshape(nrow, 1)
    Yuv = Yuv.reshape(nrow, 1)
    Zuv = Zuv.reshape(nrow, 1)
    Xvv = Xvv.reshape(nrow, 1)
    Yvv = Yvv.reshape(nrow, 1)
    Zvv = Zvv.reshape(nrow, 1)

    Xu = np.c_[Xu, Yu, Zu]
    Xv = np.c_[Xv, Yv, Zv]
    Xuu = np.c_[Xuu, Yuu, Zuu]
    Xuv = np.c_[Xuv, Yuv, Zuv]
    Xvv = np.c_[Xvv, Yvv, Zvv]

    # % First fundamental Coeffecients of the surface (E,F,G)
    E = np.einsum("ij,ij->i", Xu, Xu)
    F = np.einsum("ij,ij->i", Xu, Xv)
    G = np.einsum("ij,ij->i", Xv, Xv)

    m = np.cross(Xu, Xv, axisa=1, axisb=1)
    p = np.sqrt(np.einsum("ij,ij->i", m, m))
    n = m / np.c_[p, p, p]

    # % Second fundamental Coeffecients of the surface (L,M,N)
    L = np.einsum("ij,ij->i", Xuu, n)
    M = np.einsum("ij,ij->i", Xuv, n)
    N = np.einsum("ij,ij->i", Xvv, n)

    # % Gaussian Curvature
    K = (L * N - M**2) / (E * G - F**2)
    K = K.reshape(lr + 1, lb + 1)

    # % Mean Curvature
    H = (E * N + G * L - 2 * F * M) / (2 * (E * G - F**2))
    H = H.reshape(lr + 1, lb + 1)

    # #% Principal Curvatures
    k1 = H + np.sqrt(H**2 - K)
    k2 = H - np.sqrt(H**2 - K)

    return K, H, k1, k2  # , k1vec, k2vec
