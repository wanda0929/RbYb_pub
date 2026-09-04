#!/usr/bin/env python3
"""Physics-independent tools for the P1-1 continuous-robustness study.

The expensive gate Hamiltonian is deliberately not imported here.  The P1-1
evaluator supplies ``objective(RobustnessScenario)``; this module only constructs
nested Sobol designs, summarizes sampled minima, refines selected points locally,
and samples harmonic thermal phase space.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann, atomic_mass, hbar
from scipy.optimize import minimize
from scipy.stats import qmc


@dataclass(frozen=True)
class RobustnessScenario:
    """One point in the continuous position-and-amplitude uncertainty set."""

    delta_x_um: float
    delta_y_um: float
    delta_z_um: float
    yb_amplitude_scale: float
    rb_amplitude_scale: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.delta_x_um,
                self.delta_y_um,
                self.delta_z_um,
                self.yb_amplitude_scale,
                self.rb_amplitude_scale,
            ],
            dtype=float,
        )

    @property
    def displacement_um(self) -> np.ndarray:
        return self.as_array()[:3]

    @property
    def displacement_radius_um(self) -> float:
        return float(np.linalg.norm(self.displacement_um))


@dataclass(frozen=True)
class RobustnessDomain:
    """Bounded P1-1 domain used by both sampling and local refinement."""

    max_displacement_um: float = 0.05
    yb_amplitude_bounds: tuple[float, float] = (0.99, 1.01)
    rb_amplitude_bounds: tuple[float, float] = (0.99, 1.01)

    def contains(self, scenario: RobustnessScenario, atol: float = 1e-12) -> bool:
        return (
            scenario.displacement_radius_um <= self.max_displacement_um + atol
            and self.yb_amplitude_bounds[0] - atol
            <= scenario.yb_amplitude_scale
            <= self.yb_amplitude_bounds[1] + atol
            and self.rb_amplitude_bounds[0] - atol
            <= scenario.rb_amplitude_scale
            <= self.rb_amplitude_bounds[1] + atol
        )


DEFAULT_DOMAIN = RobustnessDomain()


def scenario_from_unit_point(
    unit_point: Sequence[float], domain: RobustnessDomain
) -> RobustnessScenario:
    """Map a point in ``[0, 1)^5`` uniformly in volume into the P1-1 domain."""

    unit_point = np.asarray(unit_point, dtype=float)
    if unit_point.shape != (5,):
        raise ValueError("unit_point must have five coordinates")
    if np.any(unit_point < 0.0) or np.any(unit_point >= 1.0):
        raise ValueError("unit_point must lie in [0, 1)^5")

    radius = domain.max_displacement_um * unit_point[0] ** (1 / 3)
    direction_cosine = 2 * unit_point[1] - 1
    azimuth = 2 * np.pi * unit_point[2]
    transverse = radius * np.sqrt(max(0.0, 1 - direction_cosine**2))

    def scale(coordinate: float, bounds: tuple[float, float]) -> float:
        return bounds[0] + coordinate * (bounds[1] - bounds[0])

    return RobustnessScenario(
        delta_x_um=float(transverse * np.cos(azimuth)),
        delta_y_um=float(transverse * np.sin(azimuth)),
        delta_z_um=float(radius * direction_cosine),
        yb_amplitude_scale=float(scale(unit_point[3], domain.yb_amplitude_bounds)),
        rb_amplitude_scale=float(scale(unit_point[4], domain.rb_amplitude_bounds)),
    )


def sobol_scenarios(
    power: int,
    domain: RobustnessDomain = DEFAULT_DOMAIN,
    *,
    scramble: bool = False,
    seed: int | None = None,
) -> tuple[RobustnessScenario, ...]:
    """Return ``2**power`` Sobol scenarios; equal settings have nested prefixes."""

    unit_points = qmc.Sobol(d=5, scramble=scramble, seed=seed).random_base2(power)
    return tuple(scenario_from_unit_point(point, domain) for point in unit_points)


@dataclass(frozen=True)
class MinimumTracePoint:
    sample_count: int
    minimum: float
    index: int


def nested_minimum_trace(
    values: Sequence[float], powers: Sequence[int] = (7, 8, 9)
) -> tuple[MinimumTracePoint, ...]:
    """Summarize minima on nested ``2**power`` prefixes of evaluated samples."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional sequence")

    trace = []
    for power in powers:
        sample_count = 2**power
        if sample_count > len(values):
            raise ValueError(f"need at least {sample_count} values for power {power}")
        index = int(np.argmin(values[:sample_count]))
        trace.append(
            MinimumTracePoint(
                sample_count=sample_count,
                minimum=float(values[index]),
                index=index,
            )
        )
    return tuple(trace)


def worst_sample_seeds(
    scenarios: Sequence[RobustnessScenario],
    values: Sequence[float],
    count: int,
) -> tuple[RobustnessScenario, ...]:
    """Select the lowest-valued sampled scenarios as local-search starts."""

    values = np.asarray(values, dtype=float)
    if len(scenarios) != len(values):
        raise ValueError("scenarios and values must have equal length")
    if count < 1 or count > len(values):
        raise ValueError("count must select at least one and no more than all samples")
    indices = np.argsort(values, kind="stable")[:count]
    return tuple(scenarios[int(index)] for index in indices)


@dataclass(frozen=True)
class LocalAdversarialResult:
    scenario: RobustnessScenario
    value: float
    success: bool
    evaluations: int
    message: str


def local_adversarial_search(
    objective: Callable[[RobustnessScenario], float],
    seeds: Sequence[RobustnessScenario],
    domain: RobustnessDomain = DEFAULT_DOMAIN,
    *,
    max_iterations: int = 200,
) -> tuple[LocalAdversarialResult, ...]:
    """Minimize a fidelity-like objective from supplied seeds inside the 5-D domain.

    This is a local SLSQP refinement, not a proof of the global minimum.  The
    intended workflow is to pass several low-fidelity Sobol points as ``seeds``
    and then re-evaluate the returned candidates with the direct pair model.
    """

    if not seeds:
        raise ValueError("at least one seed is required")
    if any(not domain.contains(seed) for seed in seeds):
        raise ValueError("all seeds must lie inside the robustness domain")

    radius = domain.max_displacement_um
    bounds = [
        (-radius, radius),
        (-radius, radius),
        (-radius, radius),
        domain.yb_amplitude_bounds,
        domain.rb_amplitude_bounds,
    ]
    constraint = {
        "type": "ineq",
        "fun": lambda point: radius**2 - float(np.dot(point[:3], point[:3])),
    }

    def scenario_from_vector(point: np.ndarray) -> RobustnessScenario:
        return RobustnessScenario(*map(float, point))

    results = []
    for seed in seeds:
        result = minimize(
            lambda point: float(objective(scenario_from_vector(point))),
            seed.as_array(),
            method="SLSQP",
            bounds=bounds,
            constraints=(constraint,),
            options={"maxiter": max_iterations, "ftol": 1e-12},
        )
        results.append(
            LocalAdversarialResult(
                scenario=scenario_from_vector(np.asarray(result.x, dtype=float)),
                value=float(result.fun),
                success=bool(result.success),
                evaluations=int(result.nfev),
                message=str(result.message),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class HarmonicThermalState:
    """Temperature and axis-resolved harmonic confinement for one isotope."""

    mass_u: float
    temperature_uk: float
    trap_frequencies_khz: tuple[float, float, float]


@dataclass(frozen=True)
class PhaseSpaceSample:
    position_um: np.ndarray
    velocity_um_per_us: np.ndarray


@dataclass(frozen=True)
class TwoSpeciesPhaseSpace:
    yb: PhaseSpaceSample
    rb: PhaseSpaceSample

    @property
    def relative_position_offset_um(self) -> np.ndarray:
        return self.yb.position_um - self.rb.position_um

    @property
    def relative_velocity_um_per_us(self) -> np.ndarray:
        return self.yb.velocity_um_per_us - self.rb.velocity_um_per_us


def harmonic_phase_space_widths(
    state: HarmonicThermalState, *, quantum: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sigma position (um) and velocity (um/us) widths per axis."""

    mass_kg = state.mass_u * atomic_mass
    temperature_k = state.temperature_uk * 1e-6
    omega = 2 * np.pi * np.asarray(state.trap_frequencies_khz, dtype=float) * 1e3
    if mass_kg <= 0 or temperature_k < 0 or np.any(omega <= 0):
        raise ValueError("mass and trap frequencies must be positive; temperature >= 0")

    if quantum:
        if temperature_k == 0:
            thermal_factor = np.ones(3)
        else:
            thermal_factor = 1 / np.tanh(hbar * omega / (2 * Boltzmann * temperature_k))
        position_variance_m2 = hbar * thermal_factor / (2 * mass_kg * omega)
        velocity_variance_m2_s2 = hbar * omega * thermal_factor / (2 * mass_kg)
    else:
        position_variance_m2 = Boltzmann * temperature_k / (mass_kg * omega**2)
        velocity_variance_m2_s2 = np.full(3, Boltzmann * temperature_k / mass_kg)

    # Numerically, 1 m/s equals 1 um/us.
    return (
        1e6 * np.sqrt(position_variance_m2),
        np.sqrt(velocity_variance_m2_s2),
    )


def sample_harmonic_phase_space(
    state: HarmonicThermalState,
    sample_count: int,
    rng: np.random.Generator,
    *,
    quantum: bool = True,
) -> PhaseSpaceSample:
    """Sample the thermal harmonic-oscillator Wigner distribution."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    position_width, velocity_width = harmonic_phase_space_widths(state, quantum=quantum)
    return PhaseSpaceSample(
        position_um=rng.normal(size=(sample_count, 3)) * position_width,
        velocity_um_per_us=rng.normal(size=(sample_count, 3)) * velocity_width,
    )


def sample_two_species_phase_space(
    yb_state: HarmonicThermalState,
    rb_state: HarmonicThermalState,
    sample_count: int,
    *,
    seed: int,
    quantum: bool = True,
) -> TwoSpeciesPhaseSpace:
    """Draw independent Yb and Rb phase-space samples with a recorded seed."""

    rng = np.random.default_rng(seed)
    return TwoSpeciesPhaseSpace(
        yb=sample_harmonic_phase_space(yb_state, sample_count, rng, quantum=quantum),
        rb=sample_harmonic_phase_space(rb_state, sample_count, rng, quantum=quantum),
    )


def rotate_phase_space(
    sample: PhaseSpaceSample, trap_to_gate_rotation: np.ndarray
) -> PhaseSpaceSample:
    """Rotate row-vector samples from trap principal axes to gate coordinates."""

    rotation = np.asarray(trap_to_gate_rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("trap_to_gate_rotation must be a 3 by 3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10):
        raise ValueError("trap_to_gate_rotation must be orthogonal")
    return PhaseSpaceSample(
        position_um=sample.position_um @ rotation.T,
        velocity_um_per_us=sample.velocity_um_per_us @ rotation.T,
    )


def ballistic_relative_trajectory(
    nominal_relative_position_um: Sequence[float],
    samples: TwoSpeciesPhaseSpace,
    times_us: Sequence[float],
) -> np.ndarray:
    """Return ``R_Yb(t)-R_Rb(t)`` while both tweezers are off.

    The result has shape ``(sample, time, xyz)``.
    """

    nominal = np.asarray(nominal_relative_position_um, dtype=float)
    times = np.asarray(times_us, dtype=float)
    if nominal.shape != (3,) or times.ndim != 1:
        raise ValueError("nominal position must be xyz and times must be one-dimensional")
    return (
        nominal[None, None, :]
        + samples.relative_position_offset_um[:, None, :]
        + samples.relative_velocity_um_per_us[:, None, :] * times[None, :, None]
    )


def doppler_detuning_mhz(
    velocity_um_per_us: np.ndarray, wavevector_rad_per_um: Sequence[float]
) -> np.ndarray:
    """Return ``-k_eff dot v / (2*pi)`` in MHz for one atom's velocity samples."""

    velocity = np.asarray(velocity_um_per_us, dtype=float)
    wavevector = np.asarray(wavevector_rad_per_um, dtype=float)
    if velocity.shape[-1] != 3 or wavevector.shape != (3,):
        raise ValueError("velocity and wavevector must end in xyz coordinates")
    return -np.einsum("...i,i->...", velocity, wavevector) / (2 * np.pi)
