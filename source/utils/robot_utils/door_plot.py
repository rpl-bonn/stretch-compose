"""
Logs and plots for door_open. Plot a saved log (or a made-up Kallax door with no argument):
    python -m utils.robot_utils.door_plot [log.json] [--urdf PATH]
"""
import argparse
import json
from dataclasses import asdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.robot_utils.door_open import Door, StretchArm, SwingPlan, footprint, plan_stance, _rot, _unit


def save_log(path, arm, door, plan, steps=(), **extra):
    """Everything needed to replay and plot a run offline."""
    with open(path, 'w') as f:
        json.dump({'door': door.to_dict(), 'plan': asdict(plan) if plan else None, 'steps': list(steps),
                   'arm': {'ext_limits': arm.ext_limits, 'yaw_limits': arm.yaw_limits}, **extra},
                  f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def plot(arm, door, plan, grid=None, steps=(), path='door_plan.png'):
    """Top-down view of the stance slack, the door and the robot along the swing, and the joints."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [3, 2]})
    if grid is not None:
        b, s = grid['bases'], np.where(np.isfinite(grid['slack']), grid['slack'], np.nan)
        m = ax.pcolormesh(b[..., 0], b[..., 1], np.ma.masked_less(s, 0), cmap='viridis', shading='auto', alpha=0.6)
        fig.colorbar(m, ax=ax, label='stance slack (m)')
    along = np.array([-door.normal[1], door.normal[0]])
    ax.plot(*np.array([door.hinge - 0.8 * along, door.hinge + 0.8 * along]).T, 'k-', lw=3, label='cabinet front')
    ax.plot(*door.hinge, 'ks', ms=8, label='hinge')
    if plan is not None:
        base = np.asarray(plan.base)
        for k in (1, -1):
            ax.add_patch(plt.Polygon(footprint(base, plan.theta[k]), fill=False, color='tab:blue', lw=2,
                                     ls='--' if k == 1 else '-'))
        colors = plt.cm.plasma(np.linspace(0, 1, len(plan.angles)))
        for k, c in enumerate(colors):
            th, yw, ex = plan.theta[k], plan.yaw[k], plan.ext[k]
            mast, wrist = base + _rot(th, arm.mast), arm.wrist_xy(base, th, ex)
            grasp = arm.grasp_xy(base, th, yw, ex)
            ax.plot(*np.array([door.hinge, door.panel_end(plan.angles[k])]).T, color=c, lw=1.5)
            ax.plot(*np.array([mast, wrist]).T, color=c, lw=3, alpha=0.7)
            ax.plot(*np.array([wrist, grasp]).T, color=c, lw=1.5, ls='--' if k == 0 else '-')
            ax.plot(*grasp, 'o', color=c, ms=4)
        ax.plot(*base, 'b+', ms=12)
        ax.arrow(*base, *(0.15 * _unit(plan.theta[1])), width=0.005, color='tab:blue')
    for st in steps:
        if 'base' in st:
            ax.plot(*st['base'], 'rx', ms=6)
        if 'grasp' in st:
            ax.plot(*st['grasp'], 'r.', ms=6)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    title = 'no feasible stance' if plan is None else (
        f"slack {plan.slack * 100:.1f} cm ({plan.limiter}), worst case {(plan.robust_slack or 0) * 100:.1f} cm\n"
        f"grip {np.degrees(plan.grip[0]):+.0f} -> {np.degrees(plan.grip[1]):+.0f} deg off square, "
        f"tracking error {plan.tracking_error * 1000:.1f} mm")
    ax.set_title(title)

    if plan is not None:
        deg = np.degrees(plan.angles)
        (e_lo, e_hi), (y_lo, y_hi) = arm.ext_limits, arm.yaw_limits
        for name, vals, lim in (('base yaw (rad)', plan.theta, None), ('wrist yaw (rad)', plan.yaw, (y_lo, y_hi)),
                                ('arm ext (m)', plan.ext, (e_lo, e_hi))):
            line, = bx.plot(deg[1:], vals[1:], 'o-', label=name)
            bx.plot(deg[0], vals[0], 'x', color=line.get_color())
            for l in lim or ():
                bx.axhline(l, color=line.get_color(), ls=':', lw=1)
        for key, mk in (('ext', 's'), ('yaw', '^'), ('theta', 'v')):
            meas = [(np.degrees(st['angle']), st['measured'][key]) for st in steps if 'measured' in st]
            if meas:
                bx.plot(*np.array(meas).T, mk, mfc='none', color='r', label=f'measured {key}')
        bx.set_xlabel('door angle (deg); x = pre-grasp')
        bx.legend(fontsize=8)
        bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', nargs='?', help='swing log JSON from a run; a made-up Kallax door if omitted')
    p.add_argument('--urdf', default='/home/ws/configs/urdf/stretch.urdf')
    p.add_argument('--hinge', choices=('left', 'right'), default='left', help='made-up door only')
    p.add_argument('--bar', action='store_true', help='made-up door only: a bar handle, not a knob')
    p.add_argument('--out', default='door_plan.png')
    args = p.parse_args()

    if args.log:
        log = json.load(open(args.log))
        arm = StretchArm(args.urdf, **log['arm'])
        door = Door(**log['door'])
        fields = SwingPlan.__dataclass_fields__
        plan = SwingPlan(**{k: v for k, v in log['plan'].items() if k in fields}) if log['plan'] else None
        steps = log.get('steps', [])
        _, grid = plan_stance(arm, door, back_off=log.get('back_off', 0.2),
                              normal_error=log.get('normal_error', np.radians(5.0)),
                              grip_tol=log.get('grip_tol', np.radians(10.0)))
    else:
        # Kallax door: front along y at x = 1 facing the robot, 33 cm wide, knob 3 cm from the edge at 0.55 m
        arm = StretchArm(args.urdf, ext_limits=(0.0, 0.5))
        side = 1.0 if args.hinge == 'left' else -1.0
        door = Door(hinge=(1.0, side * 0.165), handle=(1.0, -side * 0.135, 0.55), normal=(-1.0, 0.0))
        plan, grid = plan_stance(arm, door, back_off=0.2, grip_tol=np.radians(45.0) if args.bar else np.radians(10.0))
        steps = []
    if plan is not None:
        print(f"[door] stance {np.round(plan.base, 3)}, base turns {np.degrees(np.ptp(plan.theta[1:])):.0f} deg, "
              f"slack {plan.slack * 100:.1f} cm ({plan.limiter}), worst case {(plan.robust_slack or 0) * 100:.1f} cm, "
              f"grip {np.degrees(plan.grip[0]):+.0f} -> {np.degrees(plan.grip[1]):+.0f} deg off square")
        for a, th, yw, ex in zip(plan.angles, plan.theta, plan.yaw, plan.ext):
            print(f"  door {np.degrees(a):5.1f}  base {np.degrees(th):7.1f} deg  yaw {yw:+.2f}  ext {ex:.3f}")
    print(f"[door] plot: {plot(arm, door, plan, grid, steps, args.out)}")
