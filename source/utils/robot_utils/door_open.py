"""
Door opening for the Stretch 3: the base stands still and turns in place while arm extension and
wrist yaw keep the grasp on the handle's arc, a grid search picks where to stand.
"""
from dataclasses import dataclass
import numpy as np
import pinocchio as pin

# Base footprint corners in base_link, from base_link_collision.STL
BASE_CORNERS = np.array([[0.06, 0.18], [0.06, -0.18], [-0.29, -0.18], [-0.29, 0.18]])


def _rot(angle, v):
    """Rotate 2-D vectors v (..., 2) by angle, broadcasting."""
    c, s = np.cos(angle), np.sin(angle)
    v = np.asarray(v, dtype=float)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def _unit(angle):
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _segment_segment(a, b, c, d):
    """Distance between segments a-b and c-d, broadcasting; zero where they cross."""
    dist = np.minimum.reduce([_point_segment(a, c, d), _point_segment(b, c, d),
                              _point_segment(c, a, b), _point_segment(d, a, b)])
    crosses = ((_cross(b - a, c - a) * _cross(b - a, d - a) < 0)
               & (_cross(d - c, a - c) * _cross(d - c, b - c) < 0))
    return np.where(crosses, 0.0, dist)


class StretchArm:
    """Planar Stretch kinematics (wrist pitch and roll at zero) and limits, read off the URDF."""
    ARM_JOINTS = [f'joint_arm_l{i}' for i in range(4)]
    # Gripper folded back while the base drives, the arm just off its stop. At pitch 0 the gripper
    # touches the arm and mast from yaw 3.2 (3.4 is the factory stow, pitched down); 2.4 leaves ~6 cm
    STOW_YAW = 2.4
    RETRACTED_EXT = 0.01

    def __init__(self, urdf_path, ext_limits=None, yaw_limits=None, lift_limits=None):
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        iq = {n: model.idx_qs[model.getJointId(n)] for n in self.ARM_JOINTS + ['joint_lift', 'joint_wrist_yaw']}

        def limits(names, given):
            lo = sum(model.lowerPositionLimit[iq[n]] for n in names)
            hi = sum(model.upperPositionLimit[iq[n]] for n in names)
            return (lo, hi) if given is None else (max(lo, given[0]), min(hi, given[1]))

        self.ext_limits = limits(self.ARM_JOINTS, ext_limits)
        self.yaw_limits = limits(['joint_wrist_yaw'], yaw_limits)
        self.lift_limits = limits(['joint_lift'], lift_limits)

        def fk(ext, yaw, lift=0.5):
            q = pin.neutral(model)
            q[iq['joint_lift']] = lift
            for n in self.ARM_JOINTS:
                q[iq[n]] = ext / len(self.ARM_JOINTS)
            q[iq['joint_wrist_yaw']] = yaw
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            frame = lambda name: data.oMf[model.getFrameId(name)].translation.copy()
            return data.oMi[model.getJointId('joint_wrist_yaw')].translation.copy(), frame('link_grasp_center'), frame('link_mast')

        wrist0, grasp0, mast = fk(0.0, 0.0)
        wrist1, _, _ = fk(0.1, 0.0)
        self.yaw_origin, self.mast = wrist0[:2], mast[:2]
        self.arm_axis = (wrist1 - wrist0)[:2] / 0.1
        offset = (grasp0 - wrist0)[:2]
        self.gripper_length = float(np.linalg.norm(offset))
        self.gripper_angle = float(np.arctan2(offset[1], offset[0]))
        self.z_offset = float(grasp0[2] - 0.5)
        for ext, yaw in ((0.1, 0.8), (0.4, -1.0), (0.25, 2.5)):
            _, grasp, _ = fk(ext, yaw)
            err = np.linalg.norm(grasp[:2] - self.grasp_xy(np.zeros(2), 0.0, yaw, ext))
            assert err < 1e-3, f"planar model is off the URDF by {err:.2e} m at ext={ext}, yaw={yaw}"

    def wrist_xy(self, base, theta, ext):
        return base + _rot(theta, self.yaw_origin + np.multiply.outer(ext, self.arm_axis))

    def grasp_xy(self, base, theta, yaw, ext):
        return self.wrist_xy(base, theta, ext) + self.gripper_length * _unit(self.heading(theta, yaw))

    def heading(self, theta, yaw):
        return theta + yaw + self.gripper_angle

    def lift_for(self, z):
        return float(np.clip(z - self.z_offset, *self.lift_limits))

    def ik(self, base, grasp, heading, yaw_near=None):
        """(theta, yaw, ext) putting the grasp centre at grasp along heading; broadcasts, NaN where unreachable."""
        wrist = grasp - self.gripper_length * _unit(heading)
        to_wrist = wrist - base
        # |yaw_origin + ext * arm_axis| = |to_wrist|, the larger root
        a, o = self.arm_axis, self.yaw_origin
        aa, oa = a @ a, o @ a
        disc = oa ** 2 - aa * (o @ o - np.sum(to_wrist ** 2, -1))
        ext = (-oa + np.sqrt(np.where(disc >= 0, disc, np.nan))) / aa
        local = o + ext[..., None] * a
        theta = np.arctan2(to_wrist[..., 1], to_wrist[..., 0]) - np.arctan2(local[..., 1], local[..., 0])
        return np.unwrap(theta, axis=-1), self._continuous_yaw(_wrap(heading - theta - self.gripper_angle), yaw_near), ext

    def _continuous_yaw(self, yaw, yaw_near):
        """yaw kept continuous along the last axis, starting on the branch nearest yaw_near (mid-range if None)."""
        lo, hi = self.yaw_limits
        near = (lo + hi) / 2 if yaw_near is None else yaw_near
        yaw0 = yaw[..., :1] + 2 * np.pi * np.round((near - yaw[..., :1]) / (2 * np.pi))
        return yaw0 + np.concatenate([np.zeros_like(yaw[..., :1]), np.cumsum(_wrap(np.diff(yaw, axis=-1)), -1)], -1)


class Door:
    """A hinged door: hinge (x, y), closed handle (x, y, z) and the front's outward normal."""

    def __init__(self, hinge, handle, normal):
        self.hinge = np.asarray(hinge, dtype=float)[:2]
        self.handle = np.asarray(handle, dtype=float)[:3]
        n = np.asarray(normal, dtype=float)[:2]
        self.normal = n / np.linalg.norm(n)
        r = self.handle[:2] - self.hinge
        self.radius = float(np.linalg.norm(r))
        self.sign = 1.0 if self.normal @ (-r[1], r[0]) > 0 else -1.0
        along = r - (r @ self.normal) * self.normal
        # The panel's free edge reaches 5 cm past the handle
        self._panel = along * (1.0 + 0.05 / np.linalg.norm(along))

    @classmethod
    def from_detection(cls, selected, base, theta):
        """From a head-camera detection in base_link, the base then at (base, theta)."""
        to_fixed = lambda p: base + _rot(theta, np.asarray(p, dtype=float)[:2])
        handle = np.asarray(selected['handle_3d_base'], dtype=float)
        return cls(to_fixed(selected['hinge_3d_base']), (*to_fixed(handle), handle[2]),
                   _rot(theta, np.asarray(selected['front_normal_base'], dtype=float)[:2]))

    def moved_to(self, handle, normal=None):
        """The same door with its handle at handle, the hinge kept where it is relative to it."""
        handle = np.asarray(handle, dtype=float)
        normal = self.normal if normal is None else np.asarray(normal, dtype=float)
        turn = np.arctan2(_cross(self.normal, normal), self.normal @ normal)
        return Door(handle[:2] + _rot(turn, self.hinge - self.handle[:2]),
                    (*handle[:2], handle[2] if len(handle) > 2 else self.handle[2]), normal)

    def handle_xy(self, angle):
        return self.hinge + _rot(self.sign * np.asarray(angle), self.handle[:2] - self.hinge)

    def panel_end(self, angle):
        return self.hinge + _rot(self.sign * np.asarray(angle), self._panel)

    def angle_of(self, point):
        """How far the door is open with the handle at point."""
        a0, a1 = self.handle[:2] - self.hinge, np.asarray(point)[:2] - self.hinge
        return float(self.sign * np.arctan2(_cross(a0, a1), a0 @ a1))

    def into(self):
        """Heading pointing square into the closed door."""
        return float(np.arctan2(-self.normal[1], -self.normal[0]))

    def to_dict(self):
        return {'hinge': self.hinge.tolist(), 'handle': self.handle.tolist(), 'normal': self.normal.tolist()}


@dataclass
class SwingPlan:
    """Base stance and joints per waypoint (0 pre-grasp, 1 grasp, then the swing), with the least slack."""
    base: list
    lift: float
    angles: list
    theta: list
    yaw: list
    ext: list
    slack: float
    limiter: str
    tracking_error: float = 0.0
    robust_slack: float | None = None
    grip: tuple = (0.0, 0.0)
    pre_base: list | None = None
    pre_ext: float | None = None


def footprint(base, theta):
    """Corners of the base footprint, (..., 4, 2)."""
    return np.asarray(base)[..., None, :] + _rot(np.asarray(theta)[..., None], BASE_CORNERS)


def back_off_split(arm, theta, yaw, back_off):
    """Backing the gripper off along its heading as (base drive, arm move), broadcasting."""
    back = -back_off * _unit(np.asarray(yaw) + arm.gripper_angle)
    ax, ay = arm.arm_axis
    arm_move = back[..., 1] / ay
    return back[..., 0] - ax * arm_move, arm_move


def _slack_terms(arm, door, base, angles, theta, yaw, ext, grasp_margin, back_off=0.0, from_state=False):
    """Each constraint's slack in metres as (name, array over waypoints); negative is violated."""
    base_clear, arm_clear = 0.05, 0.04
    (e_lo, e_hi), (y_lo, y_hi) = arm.ext_limits, arm.yaw_limits
    base = np.asarray(base, dtype=float)[..., None, :]
    hinge, n = door.hinge, door.normal
    panel = door.panel_end(angles)
    mast = base + _rot(theta, arm.mast)
    wrist = arm.wrist_xy(base, theta, ext)
    ext_room = np.minimum(ext - e_lo, e_hi - ext)
    if grasp_margin:
        room_back, room_on = grasp_margin
        ext_room[..., 1] = np.minimum(ext[..., 1] - e_lo - room_back, e_hi - ext[..., 1] - room_on)

    # The base sweeps its back corners turning between waypoints: check half way too
    turning = np.concatenate([theta, (theta[..., 1:] + theta[..., :-1]) / 2], -1)
    yaw_room = arm.gripper_length * (np.minimum(yaw - y_lo, y_hi - yaw) - np.radians(5.0))
    ext_room = ext_room - 0.01
    if from_state:
        ext_room[..., :2], yaw_room[..., :2] = np.inf, np.inf
    terms = [('arm limit', ext_room),
             ('wrist yaw limit', yaw_room),
             ('base vs front', ((footprint(base, turning) - hinge) @ n).min(-1) - base_clear),
             ('arm vs front', np.minimum((mast - hinge) @ n, (wrist - hinge) @ n) - arm_clear),
             ('arm vs door', _segment_segment(mast, wrist, hinge, panel) - arm_clear)]
    if back_off:
        drive, arm_move = back_off_split(arm, theta[..., 1:2], yaw[..., 1:2], back_off)
        pre = base + drive[..., None] * _unit(theta[..., 1:2])
        pre_ext = ext[..., 1:2] + arm_move
        terms.append(('pre-grasp arm limit', np.minimum(pre_ext - e_lo, e_hi - pre_ext) - 0.01))
        # The fingertips (4 cm past the grasp centre) turning in from stowed stay clear of the front
        sweep = yaw[..., 1:2] + (arm.STOW_YAW - yaw[..., 1:2]) * np.linspace(0.0, 1.0, 9)
        wrist_in = arm.wrist_xy(pre, theta[..., 1:2], np.full(theta[..., 1:2].shape, e_lo + arm.RETRACTED_EXT))
        tips = wrist_in + (arm.gripper_length + 0.04) * _unit(arm.heading(theta[..., 1:2], sweep))
        terms.append(('gripper turning in vs front', (tips - hinge) @ n - arm_clear))
    # Door bottom taken as at most 35 cm below the handle, the base 17 cm tall
    if door.handle[2] - 0.35 < 0.17 + base_clear:
        corners = footprint(base, theta)
        edges = _segment_segment(corners, np.roll(corners, -1, axis=-2), hinge, panel[..., None, :])
        terms.append(('base vs door', edges.min(-1) - base_clear))
    return terms


def grip_values(tol, step=np.radians(15.0)):
    """Grip angles searched within tol either way, square on included."""
    return np.linspace(-tol, tol, 2 * int(np.ceil(tol / step - 1e-9)) + 1)


def grip_angle(door, heading, angle):
    """How far a gripper pointing along heading is off square on to the door open by angle."""
    return float(_wrap(heading - door.into() - door.sign * angle))


def _targets(door, grip, open_to, start=0.0, step=np.radians(10.0)):
    """Door angles, grasp-centre targets and headings: pre-grasp, grasp, then the swing from start to open_to."""
    swing = np.linspace(start, open_to, max(1, int(np.ceil((open_to - start) / step))) + 1)
    angles = np.concatenate([[start], swing])
    frac = (angles - start) / max(open_to - start, 1e-9)
    heading = door.into() + door.sign * angles + grip[0] + (grip[1] - grip[0]) * frac
    return angles, door.handle_xy(angles), heading


def plan_from(arm, door, base, grip, open_to, back_off=0.0, start=0.0, yaw_near=None, grasp_margin=None,
              from_state=False):
    """The swing with the base at base and the grip going from grip[0] to grip[1]; None if unreachable or colliding."""
    angles, grasp, heading = _targets(door, grip, open_to, start)
    theta, yaw, ext = arm.ik(np.asarray(base, float), grasp, heading, yaw_near)
    if np.isnan(ext).any():
        return None
    terms = _slack_terms(arm, door, base, angles, theta, yaw, ext, grasp_margin, back_off, from_state)
    name, worst = min(((n, float(t.min())) for n, t in terms), key=lambda t: t[1])
    if worst < 0:
        return None
    plan = SwingPlan(list(map(float, base)), arm.lift_for(door.handle[2]), angles.tolist(), theta.tolist(),
                     yaw.tolist(), ext.tolist(), worst, name, grip=(float(grip[0]), float(grip[1])))
    drive, arm_move = back_off_split(arm, theta[1], yaw[1], back_off)
    plan.pre_base = (np.asarray(base, float) + drive * _unit(theta[1])).tolist()
    plan.pre_ext = float(ext[1] + arm_move)
    plan.tracking_error = tracking_error(arm, door, plan)
    return plan


def plan_swing(arm, door, base, grip_now, grip_tol, start=0.0, yaw_near=None,
               open_angles=np.radians((100.0, 90.0, 75.0, 60.0, 45.0)), min_base_turn=0.0):
    """The swing from where the robot stands to the largest reachable door angle; None if none is."""
    for open_to in open_angles:
        if open_to <= start:
            continue
        plans = [plan_from(arm, door, base, (grip_now, g), open_to, start=start, yaw_near=yaw_near, from_state=True)
                 for g in grip_values(grip_tol)]
        plans = [p for p in plans if p is not None]
        if plans:
            turning = [p for p in plans if abs(base_turn(np.asarray(p.theta))) >= min_base_turn - 1e-6]
            return max(turning or plans, key=lambda p: p.slack - 0.05 * abs(p.grip[1]))
    return None


def door_hypotheses(door, normal_error, hinge_error):
    """The door as estimated, and with the front turned or the hinge moved either way."""
    along = np.array([-door.normal[1], door.normal[0]])
    return ([door] + [door.moved_to(door.handle, _rot(s * normal_error, door.normal)) for s in (-1, 1)]
            + [Door(door.hinge + s * hinge_error * along, door.handle, door.normal) for s in (-1, 1)])


def base_turn(theta):
    """How far the base turns from the grasp (waypoint 1) to the end of the swing."""
    return np.sum(_wrap(np.diff(theta[..., 1:], axis=-1)), axis=-1)


def plan_stance(arm, door, back_off, normal_error=np.radians(5.0), hinge_error=0.02, grip_tol=np.radians(10.0),
                base_turn_sign=-1, min_base_turn=0.0, square_grasp=True, verbose=True,
                open_angles=np.radians((100.0, 90.0, 75.0, 60.0, 45.0))):
    """
    Where to stand for the largest reachable door angle, keeping 1 cm slack on every door hypothesis.
    Returns (plan, grid), grid holding every stance's worst-case slack for plotting.
    """
    
    min_slack, grasp_margin = 0.01, (0.03, 0.08)
    along = np.array([-door.normal[1], door.normal[0]])
    u, v = np.meshgrid(np.arange(-1.0, 1.0001, 0.02), np.arange(0.15, 1.2001, 0.02), indexing='ij')
    bases = door.handle[:2] + u[..., None] * along + v[..., None] * door.normal
    doors = door_hypotheses(door, normal_error, hinge_error)
    starts = [0.0] if square_grasp else grip_values(min(grip_tol, np.radians(10.0)))
    for open_to in open_angles:
        best, best_score, slack_grid = None, -np.inf, np.full(u.shape, -np.inf)
        for grip in ((g0, g1) for g0 in starts for g1 in grip_values(grip_tol)):
            # The estimated door goes first: no hypothesis can leave more slack than it does
            slack = np.full(u.shape, np.inf)
            for d in doors:
                angles, grasp, heading = _targets(d, grip, open_to)
                heading = heading + door.into() - d.into()
                theta, yaw, ext = arm.ik(bases[..., None, :], grasp, heading)
                terms = _slack_terms(arm, d, bases, angles, theta, yaw, ext, grasp_margin, back_off)
                sl = np.nan_to_num(np.min([t.min(-1) for _, t in terms], axis=0), nan=-np.inf)
                if d is door:
                    # How the base turns is chosen on the estimated door; hypotheses only have to stay safe
                    turn = base_turn(theta)
                    turns = np.abs(turn) >= min_base_turn
                    if square_grasp:
                        turns &= np.abs(_wrap(theta[..., 1] - door.into())) <= np.radians(5.0)
                    if base_turn_sign is not None:
                        turns &= np.sign(turn) == base_turn_sign
                    sl = np.where(turns, sl, -np.inf)
                slack = np.minimum(slack, sl)
                if slack.max() < min_slack:
                    break
            slack_grid = np.maximum(slack_grid, slack)
            i = np.unravel_index(np.argmax(slack), slack.shape)
            score = slack[i] - 0.05 * (abs(grip[0]) + abs(grip[1])) / 2
            if slack[i] >= min_slack and score > best_score:
                plan = plan_from(arm, door, bases[i], grip, open_to, back_off, grasp_margin=grasp_margin)
                if plan is not None:
                    plan.robust_slack = float(slack[i])
                    best, best_score = plan, score
        if verbose:
            print(f"[door] opening to {np.degrees(open_to):.0f} deg: {int((slack_grid >= min_slack).sum())} of "
                  f"{slack_grid.size} stances keep {min_slack * 100:.0f} cm slack within "
                  f"{np.degrees(normal_error):.0f} deg / {hinge_error * 100:.0f} cm of the estimate")
        if best is not None:
            return best, {'bases': bases, 'slack': slack_grid, 'open_to': float(open_to)}
    return None, {'bases': bases, 'slack': slack_grid, 'open_to': None}


def tracking_error(arm, door, plan, samples=5):
    """Largest distance of the grasp centre from the handle's arc between swing waypoints."""
    theta, yaw, ext = (np.asarray(x) for x in (plan.theta, plan.yaw, plan.ext))
    s = np.linspace(0.0, 1.0, samples)[:, None]
    lerp = lambda x: (1 - s) * x[1:-1] + s * x[2:]
    grasp = arm.grasp_xy(np.asarray(plan.base), lerp(theta), lerp(yaw), lerp(ext))
    return float(np.abs(np.linalg.norm(grasp - door.hinge, axis=-1) - door.radius).max())
