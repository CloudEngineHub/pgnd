import numpy as np
from io import BytesIO


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float32)


def rot_mat_to_quat(rot_mat):
    w = np.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2
    x = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * w)
    y = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * w)
    z = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * w)
    return np.array([w, x, y, z], dtype=np.float32)


def save_to_splat(pts, colors, scales, quats, opacities, output_file, center=True, rotate=True):
    if center:
        pts_mean = np.mean(pts, axis=0)
        pts = pts - pts_mean
    buffer = BytesIO()
    for (v, c, s, q, o) in zip(pts, colors, scales, quats, opacities):
        position = np.array([v[0], v[1], v[2]], dtype=np.float32)
        scales = np.array([s[0], s[1], s[2]], dtype=np.float32)
        rot = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
        # SH_C0 = 0.28209479177387814
        # color = np.array([0.5 + SH_C0 * c[0], 0.5 + SH_C0 * c[1], 0.5 + SH_C0 * c[2], o[0]])
        color = np.array([c[0], c[1], c[2], o[0]])

        # rotate around x axis
        if rotate:
            rot_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
            rot_x_90 = np.linalg.inv(rot_x_90)
            position = np.dot(rot_x_90, position)
            rot = quat_mult(rot_mat_to_quat(rot_x_90), rot)

        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )
    with open(output_file, "wb") as f:
        f.write(buffer.getvalue())


def read_splat(splat_file):
    with open(splat_file, "rb") as f:
        data = f.read()
    pts = []
    colors = []
    scales = []
    quats = []
    opacities = []
    for i in range(0, len(data), 32):
        v = np.frombuffer(data[i : i + 12], dtype=np.float32)
        s = np.frombuffer(data[i + 12 : i + 24], dtype=np.float32)
        c = np.frombuffer(data[i + 24 : i + 28], dtype=np.uint8) / 255
        q = np.frombuffer(data[i + 28 : i + 32], dtype=np.uint8)
        q = (q * 1.0 - 128) / 128
        pts.append(v)
        scales.append(s)
        colors.append(c[:3])
        quats.append(q)
        opacities.append(c[3:])
    return np.array(pts), np.array(colors), np.array(scales), np.array(quats), np.array(opacities)
