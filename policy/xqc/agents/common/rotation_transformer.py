"""axis_angle <-> rotation_6d conversion, ported faithfully from pytorch3d's
own rotation_conversions.py (the exact functions the reference Diffusion
Policy implementation's own RotationTransformer('axis_angle', 'rotation_6d')
composes - diffusion_policy/model/common/rotation_transformer.py) - used for
absolute-action training (see xqc/agents/common/dataset.py's own
abs_action handling and scripts/convert_dataset_to_abs_action.py's own
docstring for why absolute actions need this at all: axis-angle has
discontinuities that hurt regression-style training, rotation_6d (Zhou et
al. 2019, "On the Continuity of Rotation Representations in Neural
Networks") doesn't).

Every function here is the direct JAX equivalent of pytorch3d's own
(same variable names, same formulas, same numerical-stability branches for
small angles) - not a reimplementation from first principles, to minimize
the chance of a subtle faithfulness gap. Reference's own composition:
forward = axis_angle_to_matrix -> matrix_to_rotation_6d,
inverse = rotation_6d_to_matrix -> matrix_to_axis_angle - see
axis_angle_to_rotation_6d/rotation_6d_to_axis_angle below for the exact
composition this module exposes.

All functions operate on the last axis (..., 3) / (..., 4) / (..., 6) /
(..., 3, 3) - batch dims are preserved, matching pytorch3d's own convention.
"""
import jax
import jax.numpy as jnp


def _sqrt_positive_part(x: jnp.ndarray) -> jnp.ndarray:
    """torch.sqrt(torch.max(0, x)) - see pytorch3d's own _sqrt_positive_part."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


def axis_angle_to_quaternion(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """(..., 3) axis-angle -> (..., 4) quaternion, real part first."""
    angles = jnp.linalg.norm(axis_angle, ord=2, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = jnp.abs(angles) < eps
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6, so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles = jnp.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        jnp.sin(half_angles) / jnp.where(small_angles, 1.0, angles),  # avoid div-by-zero in the unused branch
    )
    return jnp.concatenate([jnp.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1)


def quaternion_to_axis_angle(quaternions: jnp.ndarray) -> jnp.ndarray:
    """(..., 4) quaternion (real part first) -> (..., 3) axis-angle."""
    norms = jnp.linalg.norm(quaternions[..., 1:], ord=2, axis=-1, keepdims=True)
    half_angles = jnp.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = jnp.abs(angles) < eps
    sin_half_angles_over_angles = jnp.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        jnp.sin(half_angles) / jnp.where(small_angles, 1.0, angles),
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def quaternion_to_matrix(quaternions: jnp.ndarray) -> jnp.ndarray:
    """(..., 4) quaternion (real part first) -> (..., 3, 3) rotation matrix."""
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)

    o = jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _standardize_quaternion(quaternions: jnp.ndarray) -> jnp.ndarray:
    """Real part non-negative - see pytorch3d's own standardize_quaternion."""
    return jnp.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: jnp.ndarray) -> jnp.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion, real part first."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [
        matrix.reshape(batch_dim + (9,))[..., idx] for idx in range(9)
    ]

    q_abs = _sqrt_positive_part(
        jnp.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22],
            axis=-1,
        )
    )

    quat_by_rijk = jnp.stack(
        [
            jnp.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            jnp.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            jnp.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            jnp.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is
    # small, the candidate won't be picked (see pytorch3d's own comment).
    flr = 0.1
    quat_candidates = quat_by_rijk / (2.0 * jnp.maximum(q_abs[..., None], flr))

    # Pick the best-conditioned candidate (largest denominator), per-element.
    best = jnp.argmax(q_abs, axis=-1)
    one_hot = jax.nn.one_hot(best, num_classes=4, dtype=bool)
    out = jnp.sum(jnp.where(one_hot[..., None], quat_candidates, 0.0), axis=-2)
    return _standardize_quaternion(out)


def axis_angle_to_matrix(axis_angle: jnp.ndarray) -> jnp.ndarray:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: jnp.ndarray) -> jnp.ndarray:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix: jnp.ndarray) -> jnp.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 6), by dropping the last row
    (Zhou et al. 2019) - see pytorch3d's own matrix_to_rotation_6d."""
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def rotation_6d_to_matrix(d6: jnp.ndarray) -> jnp.ndarray:
    """(..., 6) -> (..., 3, 3) rotation matrix via Gram-Schmidt
    orthogonalization (Zhou et al. 2019, Section B) - see pytorch3d's own
    rotation_6d_to_matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / jnp.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / jnp.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = jnp.cross(b1, b2, axis=-1)
    return jnp.stack([b1, b2, b3], axis=-2)


def axis_angle_to_rotation_6d(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """(..., 3) -> (..., 6). Matches the reference's own
    RotationTransformer('axis_angle', 'rotation_6d').forward exactly (same
    two-step composition, same intermediate matrix representation)."""
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(d6: jnp.ndarray) -> jnp.ndarray:
    """(..., 6) -> (..., 3). Matches the reference's own
    RotationTransformer('axis_angle', 'rotation_6d').inverse exactly."""
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))
