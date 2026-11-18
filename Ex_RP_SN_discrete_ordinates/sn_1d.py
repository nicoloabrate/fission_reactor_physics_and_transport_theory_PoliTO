"""One-dimensional Cartesian, multigroup :math:`S_N` solver.

The module deliberately keeps the iteration levels visible rather than hiding
them in an optimised matrix formulation:

    power iteration
        outer (energy-group) iteration
            inner (within-group) iteration
                left-to-right and right-to-left diamond-difference sweeps

Both criticality and isotropic fixed-source problems use vacuum boundaries.
For a multiplying fixed-source problem, the fission cross sections may be
scaled directly or adjusted to a requested subcritical multiplication factor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


# --- aliases used in type annotations
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
EigenvalueUpdate = Literal["source_ratio", "rayleigh"]


# =============================================================================
# Convergence-error utilities
# =============================================================================
def _relative_l2(new: FloatArray, old: FloatArray) -> float:
    """Return ``||new-old||_2 / ||new||_2``, including the zero case."""

    denominator = np.linalg.norm(new)
    numerator = np.linalg.norm(new - old)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def _relative_linf(new: FloatArray, old: FloatArray) -> float:
    """Largest pointwise relative change, without division-by-zero warnings."""

    difference = np.abs(new - old)
    relative = np.empty_like(difference)
    nonzero = np.abs(new) > 0.0
    relative[nonzero] = difference[nonzero] / np.abs(new[nonzero])
    relative[~nonzero] = np.where(difference[~nonzero] == 0.0, 0.0, np.inf)
    return float(np.max(relative))


# =============================================================================
# Spatial mesh and material data
# =============================================================================
@dataclass(frozen=True)
class SlabMesh:
    """Piecewise-uniform finite-volume mesh for a slab.

    Parameters
    ----------
    region_edges:
        Coordinates of all material interfaces, including the two boundaries.
    cells_per_region:
        Number of cells in each region.
    """

    region_edges: FloatArray
    cells_per_region: IntArray
    cell_edges: FloatArray = field(init=False, repr=False)
    cell_centers: FloatArray = field(init=False)
    cell_widths: FloatArray = field(init=False, repr=False)
    region_of_cell: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # --- convert user data to arrays with a well-defined type
        region_edges = np.asarray(self.region_edges, dtype=float)
        cells_per_region = np.asarray(self.cells_per_region, dtype=int)

        # --- check consistency of the slab description
        if region_edges.ndim != 1 or cells_per_region.ndim != 1:
            raise ValueError("Mesh inputs must be one-dimensional.")
        if len(region_edges) != len(cells_per_region) + 1:
            raise ValueError("There must be one more region edge than cell count.")
        if np.any(np.diff(region_edges) <= 0.0):
            raise ValueError("Region edges must be strictly increasing.")
        if np.any(cells_per_region <= 0):
            raise ValueError("Every region must contain at least one cell.")

        # --- build a uniform mesh inside each material region
        pieces = [
                np.linspace(region_edges[r], region_edges[r + 1], n + 1)
                for r, n in enumerate(cells_per_region)
                 ]
        cell_edges = np.concatenate(
            [pieces[0]] + [piece[1:] for piece in pieces[1:]])
        region_of_cell = np.repeat(np.arange(len(cells_per_region)), cells_per_region)

        # --- dataclass is frozen: assign computed quantities explicitly
        object.__setattr__(self, "region_edges", region_edges)
        object.__setattr__(self, "cells_per_region", cells_per_region)
        object.__setattr__(self, "cell_edges", cell_edges)
        object.__setattr__(self, "cell_centers", 0.5 * (cell_edges[:-1] + cell_edges[1:]))
        object.__setattr__(self, "cell_widths", np.diff(cell_edges))
        object.__setattr__(self, "region_of_cell", region_of_cell)

    @property
    def number_of_regions(self) -> int:
        return len(self.cells_per_region)

    @property
    def number_of_cells(self) -> int:
        return len(self.cell_centers)

    @classmethod
    def from_file(cls, filename: str | Path) -> "SlabMesh":
        """Read one of the original ``geometry_*.txt`` files."""

        # The file contains R+1 region edges followed by R cell counts.
        values = np.atleast_1d(np.loadtxt(filename, comments="%", dtype=float))
        if (len(values) - 1) % 2 != 0:
            raise ValueError(f"Cannot infer the number of regions from {filename}.")
        number_of_regions = (len(values) - 1) // 2
        edges = values[:number_of_regions + 1]
        raw_cell_counts = values[number_of_regions + 1:]
        if not np.all(raw_cell_counts == raw_cell_counts.astype(int)):
            raise ValueError("Cell counts in a geometry file must be integers.")
        return cls(edges, raw_cell_counts.astype(int))


@dataclass(frozen=True)
class CrossSections:
    """Region-wise multigroup macroscopic cross sections.

    ``scatter[r, g_to, g_from]`` uses rows for destination groups and columns
    for source groups, as specified for ``S0`` in the JSON input files.
    A one-group problem is represented by arrays with one energy group.
    """

    total: FloatArray
    scatter: FloatArray
    nu_fission: FloatArray
    chi: FloatArray
    fission: FloatArray | None = None
    region_materials: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # --- standardise all cross-section containers
        total = np.asarray(self.total, dtype=float)
        scatter = np.asarray(self.scatter, dtype=float)
        nu_fission = np.asarray(self.nu_fission, dtype=float)
        chi = np.asarray(self.chi, dtype=float)
        fission = None if self.fission is None else np.asarray(self.fission, dtype=float)
        region_materials = (None if self.region_materials is None
                            else tuple(self.region_materials))

        # --- check dimensions and physical admissibility
        if total.ndim != 2:
            raise ValueError("total must have shape (regions, groups).")
        regions, groups = total.shape
        if scatter.shape != (regions, groups, groups):
            raise ValueError("scatter must have shape (regions, groups, groups).")
        if nu_fission.shape != total.shape or chi.shape != total.shape:
            raise ValueError("nu_fission and chi must have shape (regions, groups).")
        if fission is not None and fission.shape != total.shape:
            raise ValueError("fission must have shape (regions, groups).")
        if region_materials is not None and len(region_materials) != regions:
            raise ValueError("region_materials must contain one name per region.")
        arrays = {
            "total": total,
            "scatter": scatter,
            "nu_fission": nu_fission,
            "chi": chi,
        }
        if fission is not None:
            arrays["fission"] = fission
        non_finite = [name for name, values in arrays.items()
                      if not np.all(np.isfinite(values))]
        if non_finite:
            raise ValueError(f"Input data must be finite; check {non_finite}.")
        if np.any(total < 0.0):
            raise ValueError("Total cross sections must be non-negative.")
        if (np.any(scatter < 0.0) or np.any(nu_fission < 0.0)
                or np.any(chi < 0.0)
                or (fission is not None and np.any(fission < 0.0))):
            raise ValueError(
                "Cross sections, nu*Sigma_f, and chi must be non-negative.")

        # For each source group, scattering out plus fission cannot exceed
        # the total interaction cross section. A small tolerance avoids
        # rejecting otherwise consistent rounded input tables.
        removal = np.sum(scatter, axis=1)
        if fission is not None:
            removal += fission
            inconsistent_zero_fission = ((fission == 0.0)
                                          & (nu_fission != 0.0))
            if np.any(inconsistent_zero_fission):
                raise ValueError(
                    "nu*Sigma_f must be zero wherever Sigma_f is zero.")

            # nu is reconstructed only as a sanity check. Since Sigma_f and
            # nu*Sigma_f are non-negative, the inferred neutron yield must be
            # non-negative as well.
            nu = np.divide(nu_fission, fission, out=np.zeros_like(nu_fission),
                           where=fission > 0.0)
            if np.any(nu < 0.0):
                raise ValueError("The inferred fission yield nu must be non-negative.")
        tolerance = 1.0e-12 * np.maximum(1.0, total)
        if np.any(removal - total > tolerance):
            raise ValueError(
                "Scattering-out plus fission exceeds the total cross section.")

        # --- store validated arrays in the frozen dataclass
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "scatter", scatter)
        object.__setattr__(self, "nu_fission", nu_fission)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "fission", fission)
        object.__setattr__(self, "region_materials", region_materials)

    @property
    def number_of_regions(self) -> int:
        return self.total.shape[0]

    @property
    def number_of_groups(self) -> int:
        return self.total.shape[1]

    @classmethod
    def from_json(cls, filename: str | Path) -> "CrossSections":
        """Read named materials and their spatial ordering from JSON.

        The required material keys are ``Sigma_tot``, ``S0``,
        ``nuSigma_fiss`` and ``chi_tot``. ``Sigma_fiss`` is optional because
        it is not needed by the criticality iterations.
        """

        with Path(filename).open(encoding="utf-8") as stream:
            data = json.load(stream)

        # --- validate the top-level case description
        if not isinstance(data, Mapping):
            raise ValueError("The cross-section JSON root must be an object.")
        if "region_materials" not in data or "materials" not in data:
            raise ValueError("JSON input requires 'region_materials' and 'materials'.")

        region_materials = tuple(data["region_materials"])
        materials = data["materials"]
        if not region_materials:
            raise ValueError("region_materials cannot be empty.")
        if not all(isinstance(name, str) for name in region_materials):
            raise ValueError("Every entry in region_materials must be a string.")
        if not isinstance(materials, Mapping) or not materials:
            raise ValueError("materials must be a non-empty JSON object.")

        # --- infer or verify the number of energy groups
        first_name = region_materials[0]
        if first_name not in materials:
            raise ValueError(f"Material '{first_name}' is not defined.")
        first_material = materials[first_name]
        if not isinstance(first_material, Mapping) or "Sigma_tot" not in first_material:
            raise ValueError(f"Material '{first_name}' must define Sigma_tot.")
        inferred_groups = len(first_material["Sigma_tot"])
        groups = int(data.get("energy_groups", inferred_groups))
        if groups <= 0 or groups != inferred_groups:
            raise ValueError("energy_groups is inconsistent with Sigma_tot.")

        # --- expand named materials into one array row per spatial region
        total = np.empty((len(region_materials), groups))
        scatter = np.empty((len(region_materials), groups, groups))
        nu_fission = np.empty((len(region_materials), groups))
        chi = np.empty((len(region_materials), groups))
        fission_rows: list[FloatArray | None] = []

        required_keys = ("Sigma_tot", "S0", "nuSigma_fiss", "chi_tot")
        for region, material_name in enumerate(region_materials):
            if material_name not in materials:
                raise ValueError(f"Material '{material_name}' is not defined.")
            material = materials[material_name]
            if not isinstance(material, Mapping):
                raise ValueError(f"Material '{material_name}' must be a JSON object.")
            missing = [key for key in required_keys if key not in material]
            if missing:
                raise ValueError(f"Material '{material_name}' is missing {missing}.")

            total[region] = np.asarray(material["Sigma_tot"], dtype=float)
            scatter[region] = np.asarray(material["S0"], dtype=float)
            nu_fission[region] = np.asarray(material["nuSigma_fiss"], dtype=float)
            chi[region] = np.asarray(material["chi_tot"], dtype=float)

            sigma_fission = material.get("Sigma_fiss")
            fission_rows.append(None if sigma_fission is None
                                 else np.asarray(sigma_fission, dtype=float))

        # Sigma_fiss must be supplied for every material or for none of them.
        has_fission = [row is not None for row in fission_rows]
        if any(has_fission) and not all(has_fission):
            raise ValueError("Sigma_fiss must be defined for all regions or omitted for all.")
        fission = (np.stack([row for row in fission_rows if row is not None])
                   if all(has_fission) else None)

        return cls(total=total, scatter=scatter,
                   nu_fission=nu_fission, chi=chi, fission=fission,
                   region_materials=region_materials)


# =============================================================================
# Complete physical problem and numerical settings
# =============================================================================
@dataclass(frozen=True)
class SlabSNProblem:
    """A spatial mesh together with its region-wise cross sections."""

    mesh: SlabMesh
    cross_sections: CrossSections
    name: str = "slab criticality problem"

    def __post_init__(self) -> None:
        if self.mesh.number_of_regions != self.cross_sections.number_of_regions:
            raise ValueError("The geometry and cross sections have different region counts.")

    @classmethod
    def from_case(cls, directory: str | Path, case: str = "homogeneous",
                  *, multigroup: bool = False) -> "SlabSNProblem":
        """Load a named case from the input directory."""

        # --- geometry is common to the one-group and multigroup cases
        directory = Path(directory)
        mesh = SlabMesh.from_file(directory / f"geometry_{case}.txt")

        # --- select the one-group or multigroup material file
        if multigroup:
            cross_sections = CrossSections.from_json(
                directory / f"nucl_prop_MG_{case}.json")
        else:
            cross_sections = CrossSections.from_json(
                directory / f"nucl_prop_{case}.json")
        energy_description = "multigroup" if multigroup else "one-group"
        return cls(mesh, cross_sections, name=f"{case} ({energy_description})")


@dataclass(frozen=True)
class SNSettings:
    """Numerical parameters shared by one-group and multigroup calculations."""

    order: int = 2
    inner_tolerance: float = 1.0e-6
    outer_tolerance: float = 1.0e-4
    eigenvalue_tolerance: float = 1.0e-6
    max_inner_iterations: int = 10_000
    max_outer_iterations: int = 10_000
    max_power_iterations: int = 10_000
    eigenvalue_update: EigenvalueUpdate = "source_ratio"
    multigroup_warm_start: bool = True

    def __post_init__(self) -> None:
        # --- angular quadrature must contain positive/negative pairs
        if self.order <= 0 or self.order % 2:
            raise ValueError("The S_N order must be a positive even integer.")

        # --- all three nested iterations need a positive stopping tolerance
        tolerances = (
            self.inner_tolerance,
            self.outer_tolerance,
            self.eigenvalue_tolerance,
        )
        if any(tolerance <= 0.0 for tolerance in tolerances):
            raise ValueError("All convergence tolerances must be positive.")

        # --- protect every iteration level from infinite loops
        maxima = (
            self.max_inner_iterations,
            self.max_outer_iterations,
            self.max_power_iterations,
        )
        if any(maximum <= 0 for maximum in maxima):
            raise ValueError("All iteration limits must be positive.")
        if self.eigenvalue_update not in ("source_ratio", "rayleigh"):
            raise ValueError(
                "eigenvalue_update must be 'source_ratio' or 'rayleigh'.")


# =============================================================================
# Nested convergence histories and final result
# =============================================================================
@dataclass(frozen=True)
class InnerIterationHistory:
    """Convergence history of one within-group transport solve."""

    errors: tuple[float, ...]

    @property
    def iterations(self) -> int:
        return len(self.errors)

    @property
    def final_error(self) -> float:
        return self.errors[-1]


@dataclass(frozen=True)
class OuterIterationHistory:
    """One full Gauss-Seidel pass through the energy groups."""

    iteration: int
    group_flux_errors: tuple[float, ...]
    inner_iterations: tuple[InnerIterationHistory, ...]

    @property
    def error(self) -> float:
        return max(self.group_flux_errors)


@dataclass(frozen=True)
class PowerIterationHistory:
    """History associated with one fission-source (power) iteration."""

    iteration: int
    keff: float
    keff_error: float
    flux_linf_error: float
    flux_l2_error: float
    outer_iterations: tuple[OuterIterationHistory, ...]


@dataclass(frozen=True)
class SNResult:
    """Converged eigenpair and the complete nested iteration history."""

    problem: SlabSNProblem
    settings: SNSettings
    keff: float
    scalar_flux: FloatArray
    angular_flux: FloatArray
    directions: FloatArray
    weights: FloatArray
    history: tuple[PowerIterationHistory, ...]

    @property
    def converged(self) -> bool:
        return bool(self.history and
                    self.history[-1].keff_error <= self.settings.eigenvalue_tolerance)

    @property
    def power_iterations(self) -> int:
        return len(self.history)

    def normalized_scalar_flux(self, *,
                               mode: Literal["maximum", "integral"] = "maximum") -> FloatArray:
        """Return a copy of the scalar flux with a convenient normalisation."""

        # Eigenvectors have arbitrary amplitude.  Two useful conventions are
        # provided for plotting and for comparisons between calculations.
        flux = self.scalar_flux.copy()
        if mode == "maximum":
            scale = np.max(np.abs(flux))
        elif mode == "integral":
            scale = np.sum(flux * self.problem.mesh.cell_widths[None, :])
        else:
            raise ValueError("mode must be 'maximum' or 'integral'.")
        if scale == 0.0:
            raise ValueError("A zero flux cannot be normalised.")
        return flux / scale


@dataclass(frozen=True)
class FixedSourceResult:
    """Flux generated by an isotropic external source.

    ``external_source[g, i]`` is the angle-integrated source in energy group
    ``g`` and spatial cell ``i``. ``fission_scale`` multiplies every
    ``nu*Sigma_f`` value; it is zero for a nonmultiplying calculation.
    """

    problem: SlabSNProblem
    settings: SNSettings
    external_source: FloatArray
    fission_scale: float
    reference_keff: float | None
    scalar_flux: FloatArray
    angular_flux: FloatArray
    directions: FloatArray
    weights: FloatArray
    history: tuple[OuterIterationHistory, ...]

    @property
    def converged(self) -> bool:
        # The nested solvers raise RuntimeError instead of returning a result
        # when either inner or outer iterations fail to converge.
        return bool(self.history)

    @property
    def outer_iterations(self) -> int:
        """Total number of complete passes over the energy groups."""

        return len(self.history)

    @property
    def inner_iterations(self) -> int:
        """Total number of within-group inner iterations."""

        return sum(
            inner_history.iterations
            for outer_pass in self.history
            for inner_history in outer_pass.inner_iterations
        )

    def normalized_scalar_flux(self, *,
                               mode: Literal["maximum", "integral"] = "maximum"
                               ) -> FloatArray:
        """Return a normalized copy of the scalar flux."""

        flux = self.scalar_flux.copy()
        if mode == "maximum":
            scale = np.max(np.abs(flux))
        elif mode == "integral":
            scale = np.sum(flux * self.problem.mesh.cell_widths[None, :])
        else:
            raise ValueError("mode must be 'maximum' or 'integral'.")
        if scale == 0.0:
            raise ValueError("A zero flux cannot be normalised.")
        return flux / scale


# =============================================================================
# S_N criticality and fixed-source solver
# =============================================================================
class SNSolver:
    """Diamond-difference :math:`S_N` solver for a one-dimensional slab.

    The implementation uses vacuum boundary conditions at both slab surfaces.
    The same class handles one or many energy groups.
    """

    def __init__(self, problem: SlabSNProblem,
                 settings: SNSettings | None = None):
        self.problem = problem
        self.settings = settings if settings is not None else SNSettings()

        # --- generate the Gauss-Legendre angular quadrature
        # NumPy returns nodes from negative to positive.  The positive half is
        # reversed here so that the largest positive direction is swept first.
        all_directions, all_weights = np.polynomial.legendre.leggauss(self.settings.order)
        half = self.settings.order // 2
        self.directions = all_directions[half:][::-1].copy()
        self.weights = all_weights[half:][::-1].copy()

        # The angular-flux array stores positive directions first, followed by
        # their negative counterparts.  Both halves use the same weights.
        self._full_weights = np.concatenate((self.weights, self.weights))

    @classmethod
    def from_case(cls, directory: str | Path, case: str = "homogeneous", *,
                  multigroup: bool = False,
                  settings: SNSettings | None = None) -> "SNSolver":
        """Construct a solver directly from a named input case."""

        problem = SlabSNProblem.from_case(directory, case, multigroup=multigroup)
        return cls(problem, settings)

    def scalar_flux_from(self, angular_flux: FloatArray) -> FloatArray:
        """Integrate angular flux over direction.

        ``angular_flux`` may have shape ``(directions, cells)`` or
        ``(groups, directions, cells)``.
        """

        # Sum all ordinates with the Gauss-Legendre weights.  The ellipsis
        # allows the same expression to work with or without a group axis.
        return np.einsum("n,...ni->...i", self._full_weights, angular_flux)

    def isotropic_source(self, left: float, right: float, *,
                         strength: float = 1.0, group: int = 0) -> FloatArray:
        """Create a cell-wise isotropic source over ``[left, right)``.

        The returned array has shape ``(groups, cells)`` and can be passed
        directly to :meth:`solve_fixed_source`. ``strength`` is the
        angle-integrated source, so the angular transport equation receives
        ``strength / 2`` in slab geometry.
        """

        mesh = self.problem.mesh
        groups = self.problem.cross_sections.number_of_groups
        if not 0 <= group < groups:
            raise ValueError(f"group must be between 0 and {groups - 1}.")
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("strength must be finite and non-negative.")
        if not np.isfinite(left) or not np.isfinite(right) or left >= right:
            raise ValueError("Source bounds must be finite and satisfy left < right.")

        mask = (mesh.cell_centers >= left) & (mesh.cell_centers < right)
        if not np.any(mask):
            raise ValueError("The source interval does not contain any cell centres.")
        source = np.zeros((groups, mesh.number_of_cells))
        source[group, mask] = strength
        return source

    def _solve_inner_iterations(
        self, group: int, fixed_source: FloatArray,
        initial_angular_flux: FloatArray, *,
        additional_self_source: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray, InnerIterationHistory]:
        """Converge within-group scattering using repeated spatial sweeps."""

        # --- expand region-wise data onto the spatial mesh
        mesh = self.problem.mesh
        xs = self.problem.cross_sections
        region = mesh.region_of_cell
        total = xs.total[region, group]
        self_scatter = xs.scatter[region, group, group].copy()
        if additional_self_source is not None:
            self_scatter += additional_self_source

        # --- initialise the inner iteration
        angular = initial_angular_flux.copy()
        old_scalar = self.scalar_flux_from(angular)
        errors: list[float] = []
        half = self.settings.order // 2

        # =====================================================================
        # Within-group inner iteration
        # =====================================================================
        for _ in range(self.settings.max_inner_iterations):
            for direction_index, mu in enumerate(self.directions):
                # --- positive-mu sweep: left boundary -> right boundary
                # Vacuum condition: no angular flux enters from the left.
                incoming = 0.0
                for cell in range(mesh.number_of_cells):
                    streaming = 2.0 * mu / mesh.cell_widths[cell]
                    scalar_here = float(self._full_weights @ angular[:, cell])

                    # Diamond difference: solve for the cell-centred angular
                    # flux, then reconstruct the outgoing edge value below.
                    angular[direction_index, cell] = (
                        streaming * incoming
                        + 0.5 * self_scatter[cell] * scalar_here
                        + 0.5 * fixed_source[cell]
                    ) / (streaming + total[cell])
                    outgoing = 2.0 * angular[direction_index, cell] - incoming
                    incoming = outgoing

                # --- negative-mu sweep: right boundary -> left boundary
                # Only |mu| is stored; array position distinguishes -mu.
                incoming = 0.0
                negative_index = direction_index + half
                for cell in range(mesh.number_of_cells - 1, -1, -1):
                    streaming = 2.0 * mu / mesh.cell_widths[cell]
                    scalar_here = float(self._full_weights @ angular[:, cell])
                    angular[negative_index, cell] = (
                        streaming * incoming
                        + 0.5 * self_scatter[cell] * scalar_here
                        + 0.5 * fixed_source[cell]
                    ) / (streaming + total[cell])
                    outgoing = 2.0 * angular[negative_index, cell] - incoming
                    incoming = outgoing

            # --- reconstruct scalar flux and test inner convergence
            scalar = self.scalar_flux_from(angular)
            error = _relative_l2(scalar, old_scalar)
            errors.append(error)
            if error <= self.settings.inner_tolerance:
                return angular, scalar, InnerIterationHistory(tuple(errors))
            old_scalar = scalar.copy()

        raise RuntimeError(f"Inner iteration for group {group + 1} did not converge after "
                           f"{self.settings.max_inner_iterations} iterations; "
                           f"last error={errors[-1]:.3e}.")

    def _fission_source(self, scalar_flux: FloatArray, keff: float) -> FloatArray:
        """Build the fixed fission source for one power iteration."""

        # First sum production nu*Sigma_f*Phi over source groups.  Then
        # distribute the emitted neutrons over destination groups using chi.
        xs = self.problem.cross_sections
        region = self.problem.mesh.region_of_cell
        production = np.sum(xs.nu_fission[region].T * scalar_flux, axis=0)
        return xs.chi[region].T * production[None, :] / keff

    def _solve_outer_iterations(
        self, independent_source: FloatArray,
        scalar_flux: FloatArray, angular_flux: FloatArray, *,
        fission_scale: float = 0.0,
    ) -> tuple[FloatArray, FloatArray, tuple[OuterIterationHistory, ...]]:
        """Converge inter-group coupling with Gauss-Seidel outer iterations."""

        # --- initialise the outer iteration over energy groups
        xs = self.problem.cross_sections
        mesh = self.problem.mesh
        groups = xs.number_of_groups
        region = mesh.region_of_cell
        outer_history: list[OuterIterationHistory] = []

        # With one group, self-scattering is already converged by
        # _solve_inner_iterations, so a second outer pass would be redundant.
        maximum_outer_iterations = (
            1 if groups == 1 else self.settings.max_outer_iterations)

        # =====================================================================
        # Outer (thermal/up-scattering) iteration
        # =====================================================================
        for outer_iteration in range(1, maximum_outer_iterations + 1):
            errors: list[float] = []
            inner_histories: list[InnerIterationHistory] = []

            # Groups are visited sequentially.  Updated group fluxes are used
            # immediately: this is a Gauss-Seidel iteration in energy.
            for group in range(groups):
                old_group_flux = scalar_flux[group].copy()
                fixed_source = independent_source[group].copy()
                chi = xs.chi[region, group]

                # --- add scattering from all the other energy groups
                # Within-group scattering and fission are instead converged
                # inside _solve_inner_iterations.
                for source_group in range(groups):
                    if source_group == group:
                        continue
                    coupling = xs.scatter[region, group, source_group].copy()
                    coupling += (fission_scale * chi
                                 * xs.nu_fission[region, source_group])
                    fixed_source += coupling * scalar_flux[source_group]

                self_fission = (fission_scale * chi * xs.nu_fission[region, group])

                # --- choose the angular-flux initial guess
                initial_angular = angular_flux[group]
                if not self.settings.multigroup_warm_start:
                    # Restart every group solve from a unit angular flux.
                    # Keeping this selectable is useful for studying the
                    # effect of the initial guess.  This option is applied to
                    # one-group problems too, so the same switch controls the
                    # guess passed from one power iteration to the next.
                    initial_angular = np.ones_like(initial_angular)

                new_angular, new_scalar, inner_history = self._solve_inner_iterations(
                    group, fixed_source, initial_angular,
                    additional_self_source=self_fission)
                angular_flux[group] = new_angular
                scalar_flux[group] = new_scalar
                errors.append(_relative_l2(new_scalar, old_group_flux))
                inner_histories.append(inner_history)

            # --- store convergence data for this complete pass over groups
            this_history = OuterIterationHistory(
                iteration=outer_iteration, group_flux_errors=tuple(errors),
                inner_iterations=tuple(inner_histories))
            outer_history.append(this_history)

            # The outer iteration is converged only after every group has
            # changed by less than outer_tolerance.
            if groups == 1 or this_history.error <= self.settings.outer_tolerance:
                return scalar_flux, angular_flux, tuple(outer_history)

        raise RuntimeError("Outer iteration did not converge after "
                           f"{self.settings.max_outer_iterations} passes; "
                           f"last error={outer_history[-1].error:.3e}.")

    def _updated_keff(self, keff: float, old_scalar_flux: FloatArray,
                      new_scalar_flux: FloatArray) -> float:
        """Update the eigenvalue using one of the available definitions."""

        # --- calculate old and new total fission-production distributions
        xs = self.problem.cross_sections
        region = self.problem.mesh.region_of_cell
        old_production = np.sum(xs.nu_fission[region].T * old_scalar_flux, axis=0)
        new_production = np.sum(xs.nu_fission[region].T * new_scalar_flux, axis=0)

        # --- standard fission-source ratio
        if self.settings.eigenvalue_update == "source_ratio":
            denominator = np.sum(old_production)
            if denominator == 0.0:
                raise ValueError("The initial fission production is zero.")
            return float(keff * np.sum(new_production) / denominator)

        # --- Rayleigh-like update
        elif self.settings.eigenvalue_update == "rayleigh":
            if xs.number_of_groups != 1:
                raise ValueError(
                    "The 'rayleigh' update is defined here only for one-group cases.")

            # k_new = k * sum(phi_new * F phi_new) / sum(phi_new * F phi_old)
            numerator = np.sum(new_scalar_flux[0] * new_production)
            denominator = np.sum(new_scalar_flux[0] * old_production)
            if denominator == 0.0:
                raise ValueError("The Rayleigh-update denominator is zero.")
            return float(keff * numerator / denominator)

        # SNSettings validates this option at construction time. Keeping this
        # branch makes the method robust if settings are altered unexpectedly.
        else:
            raise RuntimeError(
                f"Unsupported eigenvalue update: {self.settings.eigenvalue_update!r}.")

    def solve_fixed_source(
        self,
        external_source: FloatArray,
        *,
        fission_scale: float | None = None,
        target_keff: float | None = None,
        initial_angular_flux: FloatArray | None = None,
    ) -> FixedSourceResult:
        """Solve an isotropic fixed-source transport problem.

        By default the calculation is nonmultiplying, even if the material
        data contain fission cross sections. Pass ``fission_scale=1`` to use
        the tabulated fission production directly. Alternatively,
        ``target_keff`` reproduces the MATLAB workflow: first calculate the
        unscaled eigenvalue, then multiply ``nu*Sigma_f`` by
        ``target_keff / reference_keff``. The requested value must be below
        one so that the fixed-source iteration is subcritical.
        """

        # --- validate the external source and optional fission treatment
        groups = self.problem.cross_sections.number_of_groups
        cells = self.problem.mesh.number_of_cells
        source = np.asarray(external_source, dtype=float)
        expected_source_shape = (groups, cells)
        if source.shape != expected_source_shape:
            raise ValueError(
                f"external_source has shape {source.shape}; "
                f"expected {expected_source_shape}.")
        if not np.all(np.isfinite(source)) or np.any(source < 0.0):
            raise ValueError("external_source must be finite and non-negative.")
        if fission_scale is not None and target_keff is not None:
            raise ValueError("Specify either fission_scale or target_keff, not both.")

        xs = self.problem.cross_sections
        reference_keff: float | None = None
        if target_keff is not None:
            if not 0.0 <= target_keff < 1.0:
                raise ValueError("target_keff must satisfy 0 <= target_keff < 1.")
            if not np.any(xs.nu_fission > 0.0):
                raise ValueError("target_keff requires nonzero fission production.")
            reference_keff = self.solve_critical().keff
            scale = float(target_keff / reference_keff)
        elif fission_scale is None:
            scale = 0.0
        else:
            if not np.isfinite(fission_scale) or fission_scale < 0.0:
                raise ValueError("fission_scale must be finite and non-negative.")
            scale = float(fission_scale)

        # --- a zero initial flux is the natural fixed-source initial guess
        expected_angular_shape = (groups, self.settings.order, cells)
        if initial_angular_flux is None:
            angular_flux = np.zeros(expected_angular_shape)
        else:
            angular_flux = np.asarray(initial_angular_flux, dtype=float).copy()
            if angular_flux.shape != expected_angular_shape:
                raise ValueError(
                    f"initial_angular_flux has shape {angular_flux.shape}; "
                    f"expected {expected_angular_shape}.")
            if not np.all(np.isfinite(angular_flux)):
                raise ValueError("initial_angular_flux must be finite.")
        old_scalar_flux = self.scalar_flux_from(angular_flux)

        # Scattering and fission within the same group are converged by source
        # iteration. Their cross-group contributions are converged by the
        # Gauss-Seidel outer iteration. This is the direct multigroup extension
        # of ``soln_sn_eterogeneo.m``.
        scalar_flux, angular_flux, outer_history = self._solve_outer_iterations(
            source, old_scalar_flux.copy(), angular_flux.copy(),
            fission_scale=scale)
        return FixedSourceResult(
            problem=self.problem,
            settings=self.settings,
            external_source=source.copy(),
            fission_scale=scale,
            reference_keff=reference_keff,
            scalar_flux=scalar_flux.copy(),
            angular_flux=angular_flux.copy(),
            directions=self.directions.copy(),
            weights=self.weights.copy(),
            history=outer_history,
        )

    def solve_critical(self, *, initial_keff: float = 1.0,
                       initial_angular_flux: FloatArray | None = None) -> SNResult:
        """Run power iterations and return the converged eigenpair.

        A unit angular flux is used by default and therefore gives an initial
        scalar flux equal to two.
        """

        if initial_keff <= 0.0:
            raise ValueError("initial_keff must be positive.")

        # --- initialise eigenvector and eigenvalue
        # A unit angular flux integrates to an initial scalar flux equal to 2.
        groups = self.problem.cross_sections.number_of_groups
        expected_shape = (groups, self.settings.order,
                          self.problem.mesh.number_of_cells)
        if initial_angular_flux is None:
            angular_flux = np.ones(expected_shape)
        else:
            angular_flux = np.asarray(initial_angular_flux, dtype=float).copy()
            if angular_flux.shape != expected_shape:
                raise ValueError(
                    f"initial_angular_flux has shape {angular_flux.shape}; "
                    f"expected {expected_shape}.")
        scalar_flux = self.scalar_flux_from(angular_flux)
        keff = float(initial_keff)
        history: list[PowerIterationHistory] = []

        # =====================================================================
        # Fission-source (power) iteration
        # =====================================================================
        for power_iteration in range(1, self.settings.max_power_iterations + 1):
            # --- freeze the old fission source for this power iteration
            old_scalar_flux = scalar_flux.copy()
            fission_source = self._fission_source(old_scalar_flux, keff)

            # --- converge energy-group coupling and inner iterations
            scalar_flux, angular_flux, outer_history = self._solve_outer_iterations(
                fission_source, scalar_flux.copy(), angular_flux.copy())

            # --- update k_eff and calculate convergence indicators
            new_keff = self._updated_keff(keff, old_scalar_flux, scalar_flux)
            keff_error = abs((keff - new_keff) / new_keff)
            history.append(PowerIterationHistory(
                iteration=power_iteration, keff=new_keff,
                keff_error=keff_error,
                flux_linf_error=_relative_linf(scalar_flux, old_scalar_flux),
                flux_l2_error=_relative_l2(scalar_flux, old_scalar_flux),
                outer_iterations=outer_history))
            keff = new_keff

            # --- stop when the eigenvalue error is below its tolerance
            if keff_error <= self.settings.eigenvalue_tolerance:
                return SNResult(problem=self.problem, settings=self.settings,
                                keff=keff, scalar_flux=scalar_flux.copy(),
                                angular_flux=angular_flux.copy(),
                                directions=self.directions.copy(),
                                weights=self.weights.copy(),
                                history=tuple(history))

        raise RuntimeError(
            "Power iteration did not converge after "
            f"{self.settings.max_power_iterations} iterations; "
            f"last eigenvalue error={history[-1].keff_error:.3e}.")


__all__: Sequence[str] = (
    "CrossSections",
    "FixedSourceResult",
    "InnerIterationHistory",
    "OuterIterationHistory",
    "PowerIterationHistory",
    "SlabMesh",
    "SlabSNProblem",
    "SNResult",
    "SNSettings",
    "SNSolver",
)
