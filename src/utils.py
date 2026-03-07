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