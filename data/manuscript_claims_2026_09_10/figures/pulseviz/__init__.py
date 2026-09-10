"""Small Matplotlib recipes for pulse-control papers. Arrays in, figures out."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

__version__ = '0.2.0'
COLORS = dict(blue='#0072B2', orange='#E69F00', vermillion='#D55E00',
              green='#009E73', purple='#CC79A7', skyblue='#56B4E9', gray='#666666')


MODEL_COLORS = {
    'huang': dict(blue='#0072BD', orange='#D95319', vermillion='#D95319',
                  green='#77AC30', purple='#7E2F8E', skyblue='#4DBEEE', gray='#666666'),
    'jandura': dict(blue='#1F77B4', orange='#FF7F0E', vermillion='#D62728',
                    green='#2CA02C', purple='#9467BD', skyblue='#17BECF', gray='#666666'),
}


def paper_style(model=None):
    if model is not None and model not in MODEL_COLORS:
        raise ValueError(f'Unknown model style: {model}')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.prop_cycle': cycler(color=list(COLORS.values())),
        'lines.linewidth': 1.2, 'lines.markersize': 4, 'axes.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.formatter.useoffset': False, 'savefig.dpi': 300,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.major.width': .6, 'ytick.major.width': .6,
    })
    if model is not None:
        plt.rcParams.update({
            'axes.prop_cycle': cycler(color=list(dict.fromkeys(MODEL_COLORS[model].values()))),
            'axes.spines.top': True, 'axes.spines.right': True,
            'axes.linewidth': .55, 'lines.linewidth': .9, 'lines.markersize': 3.2,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
            'xtick.major.width': .55, 'ytick.major.width': .55,
        })



def panel_label(ax, label, x=-.13, y=1.04):
    return ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
                   fontweight='normal', ha='left', va='bottom')


def _vector(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a nonempty finite vector')
    return result


def draw_curves(ax, x, curves, *, log=False):
    """curves: label -> values, or label -> dict(values=..., color=..., ls=...)."""
    x = _vector(x, 'x')
    if np.iscomplexobj(x) or np.any(np.diff(x) <= 0):
        raise ValueError('x must be real and strictly increasing')
    if not curves:
        raise ValueError('At least one curve is required')
    styles = ['-', '--', ':', '-.']
    for i, (label, curve) in enumerate(curves.items()):
        options = dict(curve) if isinstance(curve, dict) else {'values': curve}
        y = np.asarray(options.pop('values'))
        # NaNs are explicit gaps, e.g. masked phase; infinities are never plotted.
        if y.shape != x.shape or np.iscomplexobj(y) or np.any(np.isinf(y)):
            raise ValueError(f'{label}: expected real values matching x, with optional NaN gaps')
        finite = y[np.isfinite(y)]
        if not finite.size or (log and np.any(finite <= 0)):
            raise ValueError(f'{label}: log plots require positive data; every curve needs finite data')
        if 'ls' not in options and 'linestyle' not in options:
            options['linestyle'] = styles[i % len(styles)]
        ax.plot(x, y, label=label, **options)
    if log:
        ax.set_yscale('log')
    ax.legend()
    return ax


def time_stack(time, panels, *, xlabel='Time (ns)', width=7., height=None):
    """panels: [(ylabel, curves), ...]. Supply phase only when amplitudes exist."""
    if not panels:
        raise ValueError('At least one panel is required')
    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
        figsize=(width, height or 1.35 * len(panels)), layout='constrained')
    axes = axes[:, 0]
    for i, (ax, (ylabel, curves)) in enumerate(zip(axes, panels)):
        draw_curves(ax, time, curves)
        ax.set_ylabel(ylabel)
        panel_label(ax, f'({chr(97+i)})')
    axes[-1].set_xlabel(xlabel)
    return fig, axes


def sequence(ax, windows, *, xlabel='Time (ns)'):
    """windows: [(channel, start, duration, label, color), ...], in axis units."""
    if not windows:
        raise ValueError('At least one sequence window is required')
    channels = list(dict.fromkeys(w[0] for w in windows))
    for channel, start, duration, label, color in windows:
        if not np.isfinite([start, duration]).all() or duration <= 0:
            raise ValueError('Sequence starts must be finite and durations positive')
        lane = channels.index(channel)
        ax.broken_barh([(start, duration)], (lane-.3, .6), facecolors=color, alpha=.25)
        ax.text(start+duration/2, lane, label, ha='center', va='center')
    ax.set_yticks(range(len(channels)), channels)
    ax.set_ylim(len(channels)-.5, -.5)
    ax.set_xlabel(xlabel)
    return ax


def phase_trace(amplitudes, *, min_magnitude=1e-6):
    """Radians; masked near zero, unwrapped independently on valid intervals."""
    a = _vector(amplitudes, 'amplitudes')
    if not np.isfinite(min_magnitude) or min_magnitude <= 0:
        raise ValueError('min_magnitude must be positive')
    valid = np.abs(a) >= min_magnitude
    result = np.full(a.shape, np.nan)
    indices = np.flatnonzero(valid)
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        result[run] = np.unwrap(np.angle(a[run]))
    return result


def cz_phase_errors(kraus_diagonal, local_z=(0., 0.)):
    """CZ endpoint errors and invariant conditional error, in radians; order 00,01,10,11."""
    k = _vector(kraus_diagonal, 'kraus_diagonal')
    z = _vector(local_z, 'local_z')
    if k.shape != (4,) or z.shape != (2,) or np.any(abs(k) < 1e-12) or np.iscomplexobj(z):
        raise ValueError('CZ requires four nonzero amplitudes and two real local-Z angles')
    alpha, beta = z
    corrected = k * np.exp(1j*np.array([0., beta, alpha, alpha+beta]))
    residual = np.angle(corrected * [1., 1., 1., -1.] / corrected[0])
    conditional = np.angle(-k[0]*k[3]/(k[1]*k[2]))
    return residual, conditional


def robustness(scans, *, ylabel='Infidelity', width=7.):
    """scans: [(xlabel, x, curves, log_bool), ...]; no inferred error model."""
    if not scans:
        raise ValueError('At least one scan is required')
    ncols = min(2, len(scans))
    fig, axes = plt.subplots((len(scans)+ncols-1)//ncols, ncols, squeeze=False,
        figsize=(width, 2.1*((len(scans)+ncols-1)//ncols)), layout='constrained')
    for i, (xlabel, x, curves, log) in enumerate(scans):
        ax = axes.flat[i]
        draw_curves(ax, x, curves, log=log)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        panel_label(ax, f'({chr(97+i)})')
    for ax in list(axes.flat)[len(scans):]:
        ax.remove()
    return fig, list(axes.flat)[:len(scans)]


def save(fig, stem):
    path = Path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'))


def first_order_response(edges, values, *, rise_time, max_step, initial):
    """Segment-averaged first-order response; all times use the same units.

    Caller supplies every command interval, including any tail and its command.
    No tail is appended automatically.
    """
    if any(np.iscomplexobj(v) for v in (edges, values, initial)):
        raise ValueError('Response inputs must be real')
    edges = _vector(edges, 'edges').astype(float)
    values = np.asarray(values, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if (values.ndim != 2 or values.shape[0] != len(edges)-1 or
            initial.shape != (values.shape[1],) or not np.isfinite(values).all() or
            not np.isfinite(initial).all() or np.any(np.diff(edges) <= 0) or
            not np.isfinite([rise_time, max_step]).all() or rise_time <= 0 or max_step <= 0):
        raise ValueError('Response requires increasing edges, matching finite channels, and positive times')
    tau = rise_time/np.log(9)
    state = initial.copy()
    times, output = [edges[0]], []
    for duration, command in zip(np.diff(edges), values):
        steps = int(np.ceil(duration/max_step))
        dt = duration/steps
        decay = np.exp(-dt/tau)
        average = -np.expm1(-dt/tau)/(dt/tau)
        for _ in range(steps):
            output.append(command+(state-command)*average)
            state = command+(state-command)*decay
            times.append(times[-1]+dt)
    return np.asarray(times), np.asarray(output)
