import numpy as np
import casadi as ca
import numpy as np


def deg2rad(x_deg):
    return np.deg2rad(np.asarray(x_deg, dtype=float))


def rad_to_deg(x_rad):
    return np.rad2deg(np.asarray(x_rad, dtype=float))


def SafetyCheck(robot, theta_1_6, T6t=None) -> bool:
    theta = np.asarray(theta_1_6, dtype=float).reshape(6,)

    dh = robot.get_classical_dh_parameters(theta)

    a = np.asarray(dh["a"], dtype=float)
    d = np.asarray(dh["d"], dtype=float)
    alpha = dh["alpha"]
    theta = dh["theta"]

    T = np.eye(4)
    for i in range(6):
        T = T @ dh_classical_tf(a[i], deg2rad(alpha[i]), d[i], deg2rad(theta[i]))
        if T[2, 3] < 0:
            return False

    if T6t is not None:
        T_tool = T @ T6t
        if T_tool[2, 3] < 0:
            return False

    return True


def rot_x(a_rad: float) -> np.ndarray:
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    return np.array([[1, 0,  0, 0],
                     [0, ca, -sa, 0],
                     [0, sa,  ca, 0],
                     [0, 0,  0, 1]], dtype=float)


def rot_z(t_rad: float) -> np.ndarray:
    ct, st = np.cos(t_rad), np.sin(t_rad)
    return np.array([[ct, -st, 0, 0],
                     [st,  ct, 0, 0],
                     [0,    0, 1, 0],
                     [0,    0, 0, 1]], dtype=float)


def trans_x(a_m: float) -> np.ndarray:
    return np.array([[1, 0, 0, a_m],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def trans_z(d_m: float) -> np.ndarray:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d_m],
                     [0, 0, 0, 1]], dtype=float)


def rotvec_to_R(rvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=float).reshape(3,)
    th = np.linalg.norm(rvec)
    if th < 1e-12:
        return np.eye(3, dtype=float)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0]], dtype=float)
    return np.eye(3, dtype=float) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3, dtype=float)
    v = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]], dtype=float) / (2.0 * np.sin(th))
    return v * th


def pose6_to_T(pose6) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=float).reshape(6,)
    p = pose6[:3]
    rvec = pose6[3:]
    R = rotvec_to_R(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    p = T[:3, 3]
    R = T[:3, :3]
    rvec = R_to_rotvec(R)
    return np.array([p[0], p[1], p[2], rvec[0], rvec[1], rvec[2]], dtype=float)


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def dh_classical_tf(a_m: float, alpha_rad: float, d_m: float, theta_rad: float) -> np.ndarray:
    return rot_z(theta_rad) @ trans_z(d_m) @ trans_x(a_m) @ rot_x(alpha_rad)


def skew(v):
    """
    Returns the skew matrix of a vector v

    :param v: vector
    :type v: ca.MX
    :return: skew matrix of v
    :rtype: ca.MX
    """

    sk = ca.MX.zeros(3, 3)

    # Extract vector components
    x = v[0]
    y = v[1]
    z = v[2]

    sk[0, 1] = -z
    sk[1, 0] = z
    sk[0, 2] = y
    sk[2, 0] = -y
    sk[1, 2] = -x
    sk[2, 1] = x

    return sk


def xi_mat(q):
    """
    Generate the matrix for quaternion dynamics Xi,
    from Trawney's Quaternion tutorial.
    :param q: unit quaternion
    :type q: ca.MX
    :return: Xi matrix
    :rtype: ca.MX
    """
    Xi = ca.MX(4, 3)

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    # Generate Xi matrix
    Xi[0, 0] = qw
    Xi[0, 1] = -qz
    Xi[0, 2] = qy

    Xi[1, 0] = qz
    Xi[1, 1] = qw
    Xi[1, 2] = -qx

    Xi[2, 0] = -qy
    Xi[2, 1] = qx
    Xi[2, 2] = qw

    Xi[3, 0] = -qx
    Xi[3, 1] = -qy
    Xi[3, 2] = -qz

    return Xi


def q_err_mat(qr):

    mat = ca.MX.zeros((4, 4))

    q0 = qr[3]
    q1 = qr[0]
    q2 = qr[1]
    q3 = qr[2]

    mat[0, 0] = q0
    mat[0, 1] = q1
    mat[0, 2] = q2
    mat[0, 3] = q3

    mat[1, 0] = -q1
    mat[1, 1] = q0
    mat[1, 2] = q3
    mat[1, 3] = -q2

    mat[2, 0] = -q2
    mat[2, 1] = -q3
    mat[2, 2] = q0
    mat[2, 3] = q1

    mat[3, 0] = -q3
    mat[3, 1] = q2
    mat[3, 2] = -q1
    mat[3, 3] = q0

    return mat


def q_err_mat_np(qr):

    mat = np.zeros((4, 4))

    q0 = qr[3]
    q1 = qr[0]
    q2 = qr[1]
    q3 = qr[2]

    mat[0, 0] = q0
    mat[0, 1] = q1
    mat[0, 2] = q2
    mat[0, 3] = q3

    mat[1, 0] = -q1
    mat[1, 1] = q0
    mat[1, 2] = q3
    mat[1, 3] = -q2

    mat[2, 0] = -q2
    mat[2, 1] = -q3
    mat[2, 2] = q0
    mat[2, 3] = q1

    mat[3, 0] = -q3
    mat[3, 1] = q2
    mat[3, 2] = -q1
    mat[3, 3] = q0

    return mat


def r_mat_q(q):
    """
    Generate rotation matrix from unit quaternion
    :param q: unit quaternion
    :type q: ca.MX
    :return: rotation matrix, SO(3)
    :rtype: ca.MX
    """

    Rmat = ca.MX(3, 3)

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    Rmat[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    Rmat[0, 1] = 2 * qx * qy - 2 * qz * qw
    Rmat[0, 2] = 2 * qx * qz + 2 * qy * qw

    Rmat[1, 0] = 2 * qx * qy + 2 * qz * qw
    Rmat[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    Rmat[1, 2] = 2 * qy * qz - 2 * qx * qw

    Rmat[2, 0] = 2 * qx * qz - 2 * qy * qw
    Rmat[2, 1] = 2 * qy * qz + 2 * qx * qw
    Rmat[2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return Rmat


def r_mat_q_np(q):
    """
    Generate rotation matrix from unit quaternion
    :param q: unit quaternion
    :type q: ca.MX
    :return: rotation matrix, SO(3)
    :rtype: ca.MX
    """

    Rmat = np.zeros((3, 3))

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    Rmat[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    Rmat[0, 1] = 2 * qx * qy - 2 * qz * qw
    Rmat[0, 2] = 2 * qx * qz + 2 * qy * qw

    Rmat[1, 0] = 2 * qx * qy + 2 * qz * qw
    Rmat[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    Rmat[1, 2] = 2 * qy * qz - 2 * qx * qw

    Rmat[2, 0] = 2 * qx * qz - 2 * qy * qw
    Rmat[2, 1] = 2 * qy * qz + 2 * qx * qw
    Rmat[2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return Rmat


def inv_skew(sk):
    """
    Retrieve the vector from the skew-symmetric matrix.
    :param sk: skew symmetric matrix
    :type sk: ca.MX
    :return: vector corresponding to SK matrix
    :rtype: ca.MX
    """

    v = ca.MX.zeros(3, 1)

    v[0] = sk[2, 1]
    v[1] = sk[0, 2]
    v[2] = sk[1, 0]

    return v


def inv_skew_np(sk):
    """
    Retrieve the vector from the skew-symmetric matrix.
    :param sk: skew symmetric matrix
    :type sk: ca.MX
    :return: vector corresponding to SK matrix
    :rtype: ca.MX
    """

    v = np.zeros((3, 1))

    v[0] = sk[2, 1]
    v[1] = sk[0, 2]
    v[2] = sk[1, 0]

    return v


def xi_mat_np(q):
    """
    Generate the matrix for quaternion dynamics Xi,
    from Trawney's Quaternion tutorial.
    :param q: unit quaternion
    :type q: ca.MX
    :return: Xi matrix
    :rtype: ca.MX
    """
    Xi = np.zeros((4, 3))

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    # Generate Xi matrix
    Xi[0, 0] = qw
    Xi[0, 1] = -qz
    Xi[0, 2] = qy

    Xi[1, 0] = qz
    Xi[1, 1] = qw
    Xi[1, 2] = -qx

    Xi[2, 0] = -qy
    Xi[2, 1] = qx
    Xi[2, 2] = qw

    Xi[3, 0] = -qx
    Xi[3, 1] = -qy
    Xi[3, 2] = -qz

    return Xi
