"""Build the four focused teaching notebooks derived from the main tutorial."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": dedent(source).strip() + "\n"}


SETUP = code(r"""
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    working_directory = Path.cwd().resolve()
    candidates = (
        working_directory, *working_directory.parents,
        working_directory / 'Ex_RP_SN_discrete_ordinates',
        working_directory / 'RPTT_PoliTO_GitHub' / 'Ex_RP_SN_discrete_ordinates',
    )
    PROJECT_DIRECTORY = next(
        (path for path in candidates if (path / 'sn_1d.py').is_file()), None
    )
    if PROJECT_DIRECTORY is None:
        raise FileNotFoundError('Cannot find sn_1d.py.')

    sys.path.insert(0, str(PROJECT_DIRECTORY))
    from sn_1d import SNSettings, SNSolver, SlabMesh, SlabSNProblem

    DATA_DIRECTORY = PROJECT_DIRECTORY / 'data'
    plt.rcParams.update({'figure.figsize': (8, 4.5), 'axes.grid': True})
    print(f'Input: {DATA_DIRECTORY}')
""")


HELPERS = code(r"""
    def plot_inner_iteration_maps(result, selected_outer_iterations=None):
        # One dot per selected (power iteration, outer iteration) pair.
        if selected_outer_iterations is not None:
            selected_outer_iterations = set(selected_outer_iterations)
            if not selected_outer_iterations or min(selected_outer_iterations) < 1:
                raise ValueError('Select positive outer iteration numbers, or use None.')
        power_x, outer_y, per_outer, per_power = [], [], [], []
        for power in result.history:
            counts = [sum(inner.iterations for inner in outer.inner_iterations)
                      for outer in power.outer_iterations]
            total = sum(counts)
            for outer, count in zip(power.outer_iterations, counts):
                if (selected_outer_iterations is not None
                        and outer.iteration not in selected_outer_iterations):
                    continue
                power_x.append(power.iteration)
                outer_y.append(outer.iteration)
                per_outer.append(count)
                per_power.append(total)
        if not power_x:
            raise ValueError('The selected outer iterations do not occur in this result.')

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True)
        for ax, values, title, colorbar_label in (
            (axes[0], per_power, 'Total inner iterations in the power iteration on x',
             'inner iterations / power step'),
            (axes[1], per_outer, 'Inner iterations at the plotted (x, y) pair',
             'inner iterations / outer step'),
        ):
            dots = ax.scatter(power_x, outer_y, c=values, cmap='viridis',
                              s=75, edgecolors='black', linewidths=0.3)
            fig.colorbar(dots, ax=ax, label=colorbar_label)
            ax.set(xlabel='Power iteration', ylabel='Outer iteration', title=title)
            ax.set_yticks(range(1, max(outer_y) + 1))
        fig.tight_layout()
        return fig, axes


    def plot_angular_and_scalar_flux(result, group=0, normalize=True):
        # Angular and scalar flux for the selected group on the same axes.
        if not 0 <= group < result.scalar_flux.shape[0]:
            raise ValueError('GROUP_TO_PLOT is out of range.')
        x = result.problem.mesh.cell_centers
        angular = result.angular_flux[group].copy()
        scalar = result.scalar_flux[group].copy()
        if normalize:
            scale = np.max(np.abs(scalar))
            angular /= scale
            scalar /= scale
            ylabel = 'flux / maximum scalar flux of the group'
        else:
            ylabel = 'flux'
        signed_mu = np.r_[result.directions, -result.directions]
        fig, ax = plt.subplots()
        for ordinate, mu in enumerate(signed_mu):
            ax.plot(x, angular[ordinate], lw=1, alpha=0.75,
                    label=fr'$\psi(\mu={mu:+.3f})$')
        ax.plot(x, scalar, 'k', lw=2.8, label=fr'$\phi_{{{group + 1}}}$')
        for interface in result.problem.mesh.region_edges[1:-1]:
            ax.axvline(interface, color='0.6', ls='--', lw=0.8)
        ax.set(xlabel='$z$ [cm]', ylabel=ylabel,
               title=f'Angular and scalar flux — group {group + 1}')
        ax.legend(ncol=2, fontsize=8)
        return fig, ax


    def remesh_problem(problem, target_width):
        # Preserve geometry and materials while changing the spatial mesh.
        lengths = np.diff(problem.mesh.region_edges)
        cells = np.maximum(1, np.rint(lengths / target_width).astype(int))
        mesh = SlabMesh(problem.mesh.region_edges.copy(), cells)
        return SlabSNProblem(mesh, problem.cross_sections,
                             name=f'{problem.name}, dz~{target_width:g} cm')


    def keff_parameter_study(base_problem, orders, target_widths,
                             use_previous_flux=True):
        values = np.empty((len(target_widths), len(orders)))
        cell_counts = np.empty(len(target_widths), dtype=int)
        for row, width in enumerate(target_widths):
            problem = remesh_problem(base_problem, width)
            cell_counts[row] = problem.mesh.number_of_cells
            for col, order in enumerate(orders):
                settings = SNSettings(
                    order=order, inner_tolerance=1e-4, outer_tolerance=5e-4,
                    eigenvalue_tolerance=1e-4,
                    multigroup_warm_start=use_previous_flux,
                )
                values[row, col] = SNSolver(problem, settings).solve_critical().keff
        return values, cell_counts
""")


SOURCE_HELPERS = code(
    "def plot_angular_and_scalar_flux" +
    HELPERS["source"].split("def plot_angular_and_scalar_flux", 1)[1]
    .split("def remesh_problem", 1)[0]
)


WARM_START = markdown(r"""
    ## Initial guess for inner iterations

    Set `USE_PREVIOUS_FLUX_AS_INNER_GUESS` to `True` to reuse the current
    angular flux. On the first outer iteration of a power step, this is the
    angular flux from the preceding power step. Set it to `False` to restart
    every group solve from unit angular flux. In the fixed source problem,
    the same switch controls reuse between outer iterations.
""")


ANALYSIS = [
    markdown(r"""
        ## Nested iteration counts

        Each dot at `(x, y)` represents outer iteration `y` within power
        iteration `x`. Both panels use these same coordinates. On the left,
        the color is the total inner iteration count for power iteration `x`,
        summed over **all** its outer iterations and all energy groups. On
        the right, the color counts inner iterations for the **specific outer
        iteration `y`** in power iteration `x`, summed over energy groups.
        To focus on selected outer iterations, edit the tuple in the next cell;
        `None` shows every completed outer iteration. For a one group case,
        only outer iteration 1 occurs.
    """),
    code("OUTER_ITERATIONS_TO_SHOW = None  # For example, (1, 2); None shows all.\n"
         "plot_inner_iteration_maps(result, OUTER_ITERATIONS_TO_SHOW);"),
    markdown(r"""
        ## Angular and scalar flux

        Select a group with a zero based Python index. Both fluxes are divided
        by the maximum scalar flux of that group, preserving their relative
        magnitudes.
    """),
    code(r"""
        GROUP_TO_PLOT = 0
        plot_angular_and_scalar_flux(result, group=GROUP_TO_PLOT, normalize=True);
    """),
]


STUDY = [
    markdown(r"""
        ## Angular and spatial convergence of $k_{eff}$

        Both panels report $(k-k_{\mathrm{finest}})\times10^5$ pcm, where
        $k_{\mathrm{finest}}$ is computed at the largest angular order and
        smallest target mesh width in the table below. Left: change $N$ for
        each fixed mesh. Right: change the spatial mesh for each fixed $N$.
        The number of energy groups and all material data stay fixed.
    """),
    code(r"""
        # Extend these lists for a more detailed (and more expensive) study.
        ORDERS = (2, 4, 6)                    # Even angular orders
        TARGET_MESH_WIDTHS = (4.0, 2.0, 1.0)  # Target cell widths in cm

        keff_values, spatial_cells = keff_parameter_study(
            solver.problem, ORDERS, TARGET_MESH_WIDTHS,
            use_previous_flux=USE_PREVIOUS_FLUX_AS_INNER_GUESS,
        )

        reference = keff_values[np.argmin(TARGET_MESH_WIDTHS), np.argmax(ORDERS)]
        delta_pcm = (keff_values - reference) * 1e5
        print(f'Finest reference k_eff = {reference:.8f}')
        print('Rows: spatial mesh; columns: angular order N; values: delta k [pcm]')
        print('N =', ORDERS)
        for index, (width, cells) in enumerate(zip(TARGET_MESH_WIDTHS, spatial_cells)):
            print(f'{cells:3d} cells (target dz~{width:g} cm):',
                  ' '.join(f'{value:+.1f}' for value in delta_pcm[index]))

        fig, (ax_order, ax_mesh) = plt.subplots(1, 2, figsize=(12, 4.5))
        for row, cells in enumerate(spatial_cells):
            ax_order.plot(ORDERS, delta_pcm[row], 'o-', label=f'{cells} cells')
        ax_order.set(xlabel='Angular order N', ylabel=r'$(k-k_{\mathrm{finest}})\times10^5$ [pcm]',
                     title='Angular order at fixed mesh', xticks=ORDERS)
        ax_order.legend()

        for col, order in enumerate(ORDERS):
            ax_mesh.plot(spatial_cells, delta_pcm[:, col], 'o-', label=fr'$S_{order}$')
        ax_mesh.set(xlabel='Total number of cells', ylabel=r'$(k-k_{\mathrm{finest}})\times10^5$ [pcm]',
                    title='Spatial mesh at fixed angular order')
        ax_mesh.legend()
        fig.tight_layout();
    """),
]


def critical_cells(title: str, description: str, case: str, multigroup: bool,
                   order: int, inner_tol: float, outer_tol: float,
                   eigen_tol: float, group_to_plot: int = 0) -> list[dict]:
    return [
        markdown(f"# {title}\n\n{description}"), SETUP, HELPERS, WARM_START,
        code(f"""
            USE_PREVIOUS_FLUX_AS_INNER_GUESS = True

            settings = SNSettings(
                order={order}, inner_tolerance={inner_tol!r},
                outer_tolerance={outer_tol!r}, eigenvalue_tolerance={eigen_tol!r},
                multigroup_warm_start=USE_PREVIOUS_FLUX_AS_INNER_GUESS,
            )
            solver = SNSolver.from_case(
                DATA_DIRECTORY, case={case!r}, multigroup={multigroup!r}, settings=settings
            )
            result = solver.solve_critical()

            print(f'k_eff = {{result.keff:.8f}}')
            print(f'power iterations = {{result.power_iterations}}')
            print(f'groups = {{result.scalar_flux.shape[0]}}')
            print(f'materials = {{result.problem.cross_sections.region_materials}}')
        """),
        *ANALYSIS[:-1],
        code(f"GROUP_TO_PLOT = {group_to_plot}\nplot_angular_and_scalar_flux(result, GROUP_TO_PLOT);"),
        *STUDY,
    ]


source_cells = [
    markdown(r"""
        # Ex1_localized_isotropic_source

        A nonmultiplying four group slab. The source is isotropic in angle,
        occupies only the chosen energy group, and is distributed across the
        two central cells to preserve symmetry. `TOTAL_SOURCE` is its
        spatially integrated strength.
    """),
    SETUP, SOURCE_HELPERS, WARM_START,
    code(r"""
        USE_PREVIOUS_FLUX_AS_INNER_GUESS = True
        SOURCE_GROUP = 0       # Zero based group index: 0, ..., G-1
        TOTAL_SOURCE = 1.0

        settings = SNSettings(
            order=4, inner_tolerance=1e-6, outer_tolerance=1e-5,
            multigroup_warm_start=USE_PREVIOUS_FLUX_AS_INNER_GUESS,
        )
        solver = SNSolver.from_case(
            DATA_DIRECTORY, case='homogeneous', multigroup=True, settings=settings
        )

        mesh = solver.problem.mesh
        slab_center = 0.5 * (mesh.region_edges[0] + mesh.region_edges[-1])
        distance = np.abs(mesh.cell_centers - slab_center)
        central_cells = np.isclose(distance, distance.min())
        external_source = np.zeros((solver.problem.cross_sections.number_of_groups,
                                    mesh.number_of_cells))
        external_source[SOURCE_GROUP, central_cells] = (
            TOTAL_SOURCE / mesh.cell_widths[central_cells].sum()
        )
        result = solver.solve_fixed_source(external_source)

        print(f'source cells = {np.flatnonzero(central_cells).tolist()}')
        print(f'source cell centers = {mesh.cell_centers[central_cells]} cm')
        print(f'spatial integral = '
              f'{np.sum(external_source[SOURCE_GROUP] * mesh.cell_widths):.6f}')
        print(f'outer iterations = {result.outer_iterations}')
    """),
    markdown(r"""
        The source profile identifies its position and energy group.
    """),
    code(r"""
        fig, ax = plt.subplots()
        ax.step(mesh.cell_centers, external_source[SOURCE_GROUP], where='mid')
        ax.set(xlabel='$z$ [cm]', ylabel='$Q_g(z)$',
               title=f'External source in group {SOURCE_GROUP + 1}');
    """),
    markdown(r"""
        ## Fixed source scalar flux and outer iterations

        The shaded interval marks the localized source, as in the fixed source
        example in the original tutorial. The second panel shows inner solves
        per outer iteration, summed over all energy groups. A fixed source
        problem has no power iterations or $k_{eff}$.
    """),
    code(r"""
        fig, (ax_flux, ax_iterations) = plt.subplots(1, 2, figsize=(12, 4.5))
        for group, flux in enumerate(result.scalar_flux, start=1):
            ax_flux.plot(mesh.cell_centers, flux, label=f'Group {group}')
        ax_flux.axvspan(mesh.cell_edges[np.flatnonzero(central_cells)[0]],
                        mesh.cell_edges[np.flatnonzero(central_cells)[-1] + 1],
                        color='tab:orange', alpha=0.2, label='External source')
        ax_flux.set(xlabel='$z$ [cm]', ylabel='Scalar flux',
                    title='Fixed source solution')
        ax_flux.legend()

        outer_counts = [sum(inner.iterations for inner in outer.inner_iterations)
                        for outer in result.history]
        ax_iterations.plot(range(1, len(outer_counts) + 1), outer_counts, 'o-')
        ax_iterations.set(xlabel='Outer iteration',
                          ylabel='Inner iterations across all groups',
                          title='Fixed source iteration cost')
        ax_iterations.set_xticks(range(1, len(outer_counts) + 1))
        fig.tight_layout();
    """),
    ANALYSIS[2], ANALYSIS[3],
]


notebooks = {
    'Ex1_localized_isotropic_source.ipynb': source_cells,
    'Ex2_homogeneous_one_group.ipynb': critical_cells(
        'Ex2_homogeneous_one_group',
        'A homogeneous one group slab with vacuum boundary conditions.',
        'homogeneous', False, 4, 1e-6, 1e-4, 1e-6),
    'Ex3_homogeneous_multigroup.ipynb': critical_cells(
        'Ex3_homogeneous_multigroup',
        'A homogeneous four group slab with energy coupling and power iterations.',
        'homogeneous', True, 4, 1e-5, 1e-4, 1e-5, group_to_plot=3),
    'Ex4_five_region_multigroup.ipynb': critical_cells(
        'Ex4_five_region_multigroup',
        ('Symmetric reflector–outer fuel–inner fuel–outer fuel–reflector geometry. '
         'The educational four group data use homogeneous fuel for the outer fuel '
         'and increase nuSigma_f by 50% in the inner fuel, following the contrast '
         'of the existing one group five region case.'),
        '5regions', True, 4, 1e-5, 1e-4, 1e-5, group_to_plot=3),
}


metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

for filename, cells in notebooks.items():
    cells_with_ids = [dict(cell, id=f"cell-{index:02d}")
                      for index, cell in enumerate(cells)]
    notebook = {"cells": cells_with_ids, "metadata": metadata,
                "nbformat": 4, "nbformat_minor": 5}
    (HERE / filename).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    print(f'Created {filename} ({len(cells)} cells)')
