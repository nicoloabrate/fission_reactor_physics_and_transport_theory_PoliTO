"""Small regression tests for the teaching-oriented S_N solver."""

import json
from pathlib import Path

import numpy as np
import pytest

from sn_1d import CrossSections, SNSettings, SNSolver, SlabSNProblem

DATA_DIRECTORY = Path(__file__).parent / "data"


def test_input_files_are_loaded_with_expected_shapes() -> None:
    one_group = SlabSNProblem.from_case(DATA_DIRECTORY,
                                        "reflected",
                                        multigroup=False)
    multigroup = SlabSNProblem.from_case(DATA_DIRECTORY,
                                         "reflected",
                                         multigroup=True)

    assert one_group.mesh.number_of_regions == 3
    assert one_group.mesh.number_of_cells == 100
    assert one_group.cross_sections.total.shape == (3, 1)
    assert multigroup.cross_sections.total.shape == (3, 4)
    assert multigroup.cross_sections.scatter.shape == (3, 4, 4)
    assert one_group.cross_sections.region_materials == ("reflector", "fuel", "reflector")
    assert multigroup.cross_sections.region_materials == ("reflector", "fuel", "reflector")


def test_all_cross_section_json_files_are_self_consistent() -> None:
    filenames = sorted(DATA_DIRECTORY.glob("nucl_prop*.json"))
    assert len(filenames) == 6

    for filename in filenames:
        with filename.open(encoding="utf-8") as stream:
            raw_data = json.load(stream)

        cross_sections = CrossSections.from_json(filename)
        number_of_regions = len(raw_data["region_materials"])
        number_of_groups = raw_data["energy_groups"]

        assert cross_sections.number_of_regions == number_of_regions
        assert cross_sections.number_of_groups == number_of_groups
        assert cross_sections.scatter.shape == (
            number_of_regions, number_of_groups, number_of_groups)
        assert set(raw_data["region_materials"]) <= set(raw_data["materials"])


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("total", np.nan),
        ("total", -0.1),
        ("scatter", -0.1),
        ("nu_fission", -0.1),
        ("chi", -0.1),
        ("fission", -0.1),
    ],
)
def test_cross_sections_reject_nonfinite_or_negative_data(
    field: str, invalid_value: float,
) -> None:
    inputs = {
        "total": np.array([[1.0]]),
        "scatter": np.array([[[0.2]]]),
        "nu_fission": np.array([[0.5]]),
        "chi": np.array([[1.0]]),
        "fission": np.array([[0.2]]),
    }
    inputs[field].flat[0] = invalid_value

    with pytest.raises(ValueError):
        CrossSections(**inputs)


def test_cross_sections_reject_inconsistent_fission_yield() -> None:
    with pytest.raises(ValueError, match=r"nu\*Sigma_f must be zero"):
        CrossSections(
            total=np.array([[1.0]]),
            scatter=np.array([[[0.2]]]),
            nu_fission=np.array([[0.5]]),
            chi=np.array([[1.0]]),
            fission=np.array([[0.0]]),
        )


def test_cross_sections_reject_removal_larger_than_total() -> None:
    with pytest.raises(ValueError, match="exceeds the total"):
        CrossSections(
            total=np.array([[1.0]]),
            scatter=np.array([[[0.9]]]),
            nu_fission=np.array([[0.5]]),
            chi=np.array([[1.0]]),
            fission=np.array([[0.2]]),
        )


def test_quadrature_integrates_constant_over_all_directions() -> None:
    solver = SNSolver.from_case(
        DATA_DIRECTORY,
        "homogeneous",
        settings=SNSettings(order=8),
    )

    assert np.isclose(2.0 * np.sum(solver.weights), 2.0)
    assert np.all(solver.directions > 0.0)


def test_power_iteration_default_tolerances() -> None:
    settings = SNSettings()

    assert settings.eigenvalue_tolerance == 1.0e-8
    assert settings.power_flux_tolerance == 1.0e-7


@pytest.mark.parametrize("invalid_tolerance", [0.0, -1.0e-7])
def test_power_flux_tolerance_must_be_positive(invalid_tolerance: float) -> None:
    with pytest.raises(ValueError, match="convergence tolerances"):
        SNSettings(power_flux_tolerance=invalid_tolerance)


def test_one_group_homogeneous_regression() -> None:
    settings = SNSettings(order=2,
                          inner_tolerance=1.0e-6,
                          eigenvalue_tolerance=1.0e-6)
    result = SNSolver.from_case(
        DATA_DIRECTORY,
        "homogeneous",
        multigroup=False,
        settings=settings,
    ).solve_critical()

    assert result.converged
    assert result.history[-1].keff_error <= settings.eigenvalue_tolerance
    assert result.history[-1].flux_l2_error <= settings.power_flux_tolerance
    assert any(
        step.keff_error <= settings.eigenvalue_tolerance
        and step.flux_l2_error > settings.power_flux_tolerance
        for step in result.history[:-1]
    )
    assert result.power_iterations == 392
    assert np.isclose(result.keff, 1.2148446687, rtol=0.0, atol=5.0e-10)
    assert result.scalar_flux.shape == (1, 100)


def test_rayleigh_update_regression() -> None:
    settings = SNSettings(
        order=2,
        inner_tolerance=1.0e-6,
        eigenvalue_tolerance=1.0e-6,
        eigenvalue_update="rayleigh",
    )
    result = SNSolver.from_case(
        DATA_DIRECTORY,
        "homogeneous",
        multigroup=False,
        settings=settings,
    ).solve_critical()

    assert result.converged
    assert result.history[-1].keff_error <= settings.eigenvalue_tolerance
    assert result.history[-1].flux_l2_error <= settings.power_flux_tolerance
    assert result.power_iterations == 392
    assert np.isclose(result.keff, 1.2148454245, rtol=0.0, atol=5.0e-10)


def test_multigroup_homogeneous_regression() -> None:
    settings = SNSettings(
        order=2,
        inner_tolerance=1.0e-5,
        outer_tolerance=1.0e-4,
        eigenvalue_tolerance=1.0e-5,
        multigroup_warm_start=False,
    )
    result = SNSolver.from_case(
        DATA_DIRECTORY,
        "homogeneous",
        multigroup=True,
        settings=settings,
    ).solve_critical()

    assert result.converged
    assert result.history[-1].keff_error <= settings.eigenvalue_tolerance
    assert result.history[-1].flux_l2_error <= settings.power_flux_tolerance
    assert result.power_iterations == 83
    assert np.isclose(result.keff, 1.2053945251, rtol=0.0, atol=5.0e-10)
    assert result.scalar_flux.shape == (4, 100)
    assert len(result.history[0].outer_iterations) > 1


def test_warm_start_switch_also_controls_one_group_inner_guess() -> None:
    common = dict(order=2, inner_tolerance=1.0e-6,
                  eigenvalue_tolerance=1.0e-4)
    warm = SNSolver.from_case(
        DATA_DIRECTORY, "homogeneous", multigroup=False,
        settings=SNSettings(**common, multigroup_warm_start=True),
    ).solve_critical()
    cold = SNSolver.from_case(
        DATA_DIRECTORY, "homogeneous", multigroup=False,
        settings=SNSettings(**common, multigroup_warm_start=False),
    ).solve_critical()

    warm_counts = [
        power.outer_iterations[0].inner_iterations[0].iterations
        for power in warm.history
    ]
    cold_counts = [
        power.outer_iterations[0].inner_iterations[0].iterations
        for power in cold.history
    ]
    assert warm_counts != cold_counts
    assert np.isclose(warm.keff, cold.keff, rtol=0.0, atol=2.0e-4)


def test_isotropic_source_uses_cell_centres_and_selected_group() -> None:
    solver = SNSolver.from_case(DATA_DIRECTORY, "homogeneous", multigroup=True)
    source = solver.isotropic_source(30.0, 70.0, strength=2.5, group=2)

    assert source.shape == (4, 100)
    assert np.count_nonzero(source[2]) == 40
    assert np.all(source[2, 30:70] == 2.5)
    assert np.count_nonzero(np.delete(source, 2, axis=0)) == 0


def test_nonmultiplying_fixed_source_regression() -> None:
    settings = SNSettings(order=2, inner_tolerance=1.0e-6)
    solver = SNSolver.from_case(
        DATA_DIRECTORY, "homogeneous", multigroup=False, settings=settings)
    source = solver.isotropic_source(30.0, 70.0)
    result = solver.solve_fixed_source(source)

    assert result.converged
    assert result.outer_iterations == 1
    assert result.inner_iterations == 186
    assert result.fission_scale == 0.0
    assert result.reference_keff is None
    assert result.scalar_flux.shape == (1, 100)
    assert np.allclose(result.scalar_flux, result.scalar_flux[:, ::-1], atol=1.0e-6)
    assert np.isclose(np.max(result.scalar_flux), 34.7967221976,
                      rtol=0.0, atol=5.0e-9)


def test_fixed_source_can_scale_fission_to_target_keff() -> None:
    settings = SNSettings(
        order=2,
        inner_tolerance=1.0e-6,
        eigenvalue_tolerance=1.0e-6,
    )
    solver = SNSolver.from_case(
        DATA_DIRECTORY, "homogeneous", multigroup=False, settings=settings)
    source = solver.isotropic_source(30.0, 70.0)
    result = solver.solve_fixed_source(source, target_keff=0.95)

    assert result.converged
    assert result.reference_keff is not None
    assert np.isclose(result.reference_keff, 1.2148446687,
                      rtol=0.0, atol=5.0e-10)
    assert np.isclose(result.fission_scale,
                      0.95 / result.reference_keff,
                      rtol=0.0, atol=1.0e-15)
    assert result.outer_iterations == 1
    assert result.inner_iterations == 2758
    assert np.isclose(np.max(result.scalar_flux), 579.8785591489,
                      rtol=0.0, atol=5.0e-8)


def test_multigroup_fixed_source_transfers_neutrons_between_groups() -> None:
    settings = SNSettings(order=2, inner_tolerance=1.0e-5,
                          outer_tolerance=1.0e-4)
    solver = SNSolver.from_case(
        DATA_DIRECTORY, "homogeneous", multigroup=True, settings=settings)
    scatter_before = solver.problem.cross_sections.scatter.copy()
    source = solver.isotropic_source(30.0, 70.0, group=0)
    result = solver.solve_fixed_source(source)

    assert result.converged
    assert result.scalar_flux.shape == (4, 100)
    assert np.all(np.max(result.scalar_flux, axis=1) > 0.0)
    assert len(result.history) == 3
    assert np.array_equal(solver.problem.cross_sections.scatter, scatter_before)
