"""Build the four focused teaching notebooks derived from the main tutorial."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent


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

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

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
    def _inner_counts_at_power(result, power_frame):
        # power_frame is deliberately zero based, as in an animation timeline.
        if not 0 <= power_frame < len(result.history):
            raise ValueError(
                f'power_frame must be between 0 and {len(result.history) - 1}.'
            )
        power = result.history[power_frame]
        outer_numbers = np.array(
            [outer.iteration for outer in power.outer_iterations], dtype=int
        )
        group_counts = np.array([
            [inner.iterations for inner in outer.inner_iterations]
            for outer in power.outer_iterations
        ], dtype=int)
        return power, outer_numbers, group_counts


    def plot_inner_vs_outer(result, power_frame=0, show_groups=True, y_max=None):
        # Inner iterations versus outer iteration at one selected power frame.
        power, outer_numbers, group_counts = _inner_counts_at_power(
            result, power_frame
        )
        totals = group_counts.sum(axis=1)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(outer_numbers, totals, 'ko-', lw=2.2,
                label='Total over groups')
        if show_groups:
            for group, counts in enumerate(group_counts.T, start=1):
                ax.plot(outer_numbers, counts, 'o--', lw=1.2,
                        label=f'Group {group}')
        ax.set(
            xlabel='Outer iteration',
            ylabel='Number of inner iterations',
            title=(f'Power frame {power_frame} / {len(result.history) - 1} '
                   f'(solver iteration {power.iteration}, '
                   f'$k_{{eff}}={power.keff:.7f}$)'),
            xticks=outer_numbers,
        )
        ax.set_ylim(bottom=0, top=y_max)
        ax.legend()
        fig.tight_layout()
        return fig, ax


    def inner_iteration_widget(result, initial_frame=0, interval=600,
                               repeat=True, show_groups=True, y_max=None):
        # Linked Play and slider controls animate all recorded power frames.
        last_frame = len(result.history) - 1
        if not 0 <= initial_frame <= last_frame:
            raise ValueError(f'initial_frame must be between 0 and {last_frame}.')
        global_y_max = 1.05 * max(
            sum(inner.iterations for inner in outer.inner_iterations)
            for power in result.history
            for outer in power.outer_iterations
        )

        play = widgets.Play(
            value=initial_frame, min=0, max=last_frame, step=1,
            interval=interval, description='Play', repeat=repeat,
        )
        frame = widgets.IntSlider(
            value=initial_frame, min=0, max=last_frame, step=1,
            description='Power frame', continuous_update=False,
            readout=True, layout=widgets.Layout(width='520px'),
        )
        speed = widgets.IntSlider(
            value=interval, min=100, max=2000, step=100,
            description='Frame [ms]', continuous_update=False,
            layout=widgets.Layout(width='360px'),
        )
        repeat_control = widgets.Checkbox(value=repeat, description='Repeat')
        groups_control = widgets.Checkbox(
            value=show_groups, description='Show single groups'
        )
        y_limit = widgets.FloatText(
            value=0.0 if y_max is None else y_max,
            description='Y max (0=global)',
            layout=widgets.Layout(width='220px'),
        )

        widgets.jslink((play, 'value'), (frame, 'value'))
        widgets.jslink((speed, 'value'), (play, 'interval'))
        widgets.jslink((repeat_control, 'value'), (play, 'repeat'))

        def redraw(power_frame, show_single_groups, requested_y_max):
            upper_limit = (global_y_max if requested_y_max <= 0
                           else requested_y_max)
            plot_inner_vs_outer(
                result, power_frame=power_frame,
                show_groups=show_single_groups, y_max=upper_limit,
            )
            plt.show()

        output = widgets.interactive_output(
            redraw,
            {
                'power_frame': frame,
                'show_single_groups': groups_control,
                'requested_y_max': y_limit,
            },
        )
        controls = widgets.VBox([
            widgets.HBox([play, frame]),
            widgets.HBox([speed, repeat_control, groups_control, y_limit]),
        ])
        return widgets.VBox([controls, output])


    def plot_angular_and_scalar_flux(result, group=0, direction='all',
                                     normalize=True):
        # Select one group and either scalar, all angular, or one direction.
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
        if direction == 'all':
            for ordinate, mu in enumerate(signed_mu):
                ax.plot(x, angular[ordinate], lw=1, alpha=0.75,
                        label=fr'$\psi(\mu={mu:+.3f})$')
        elif direction != 'scalar':
            if not isinstance(direction, (int, np.integer)):
                raise ValueError("direction must be 'all', 'scalar', or an index.")
            if not 0 <= direction < len(signed_mu):
                raise ValueError('The direction index is out of range.')
            mu = signed_mu[direction]
            ax.plot(x, angular[direction], lw=2,
                    label=fr'$\psi(\mu={mu:+.3f})$')
        ax.plot(x, scalar, 'k', lw=2.8,
                ls='--' if direction not in ('all', 'scalar') else '-',
                label=fr'$\phi_{{{group + 1}}}$')
        for interface in result.problem.mesh.region_edges[1:-1]:
            ax.axvline(interface, color='0.6', ls='--', lw=0.8)
        ax.set(xlabel='$z$ [cm]', ylabel=ylabel,
               title=f'Flux — group {group + 1}')
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        return fig, ax


    def flux_widget(result, initial_group=0, initial_direction='all',
                    normalize=True):
        # Interactive group/direction selection without rerunning the solver.
        groups = result.scalar_flux.shape[0]
        signed_mu = np.r_[result.directions, -result.directions]
        direction_options = [('All angular directions', 'all'),
                             ('Scalar flux only', 'scalar')]
        direction_options.extend(
            (f'Direction {index}: mu={mu:+.5f}', index)
            for index, mu in enumerate(signed_mu)
        )
        valid_directions = {value for _, value in direction_options}
        if not 0 <= initial_group < groups:
            raise ValueError(f'initial_group must be between 0 and {groups - 1}.')
        if initial_direction not in valid_directions:
            raise ValueError('initial_direction is not available.')

        group_control = widgets.Dropdown(
            options=[(f'Group {group + 1}', group) for group in range(groups)],
            value=initial_group, description='Energy group',
        )
        direction_control = widgets.Dropdown(
            options=direction_options, value=initial_direction,
            description='Direction',
            layout=widgets.Layout(width='310px'),
        )
        normalize_control = widgets.Checkbox(
            value=normalize, description='Normalize by max scalar flux'
        )

        def redraw(group, direction, normalize_flux):
            plot_angular_and_scalar_flux(
                result, group=group, direction=direction,
                normalize=normalize_flux,
            )
            plt.show()

        output = widgets.interactive_output(
            redraw,
            {
                'group': group_control,
                'direction': direction_control,
                'normalize_flux': normalize_control,
            },
        )
        controls = widgets.HBox(
            [group_control, direction_control, normalize_control]
        )
        return widgets.VBox([controls, output])


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
        ## Inner versus outer iterations at one power step

        `POWER_FRAME` is the zero-based position in the recorded power-method
        history: frame 0 is the first completed power iteration and the last
        frame is `result.power_iterations - 1`. The black curve is the total
        number of inner iterations across all groups; the dashed curves show
        the individual group contributions. Change the parameters below and
        rerun only this cell to inspect one frame precisely.
    """),
    code(r"""
        POWER_FRAME = 0
        SHOW_GROUP_COUNTS = True
        INNER_Y_MAX = None  # Use None for automatic scaling

        plot_inner_vs_outer(
            result,
            power_frame=POWER_FRAME,
            show_groups=SHOW_GROUP_COUNTS,
            y_max=INNER_Y_MAX,
        );
    """),
    markdown(r"""
        ## Continuous animation of the nested iterations

        Press play to move continuously from power frame 0 to the final
        recorded frame, or drag the slider to any frame. The speed, repeat
        mode, group curves, and vertical scale can also be changed directly
        in the controls. `ANIMATION_INTERVAL_MS` is the initial delay between
        frames.
    """),
    code(r"""
        ANIMATION_INTERVAL_MS = 600
        ANIMATION_REPEAT = True

        iteration_explorer = inner_iteration_widget(
            result,
            initial_frame=POWER_FRAME,
            interval=ANIMATION_INTERVAL_MS,
            repeat=ANIMATION_REPEAT,
            show_groups=SHOW_GROUP_COUNTS,
            y_max=INNER_Y_MAX,
        )
        display(iteration_explorer)
    """),
    markdown(r"""
        ## Interactive angular and scalar flux

        Select the energy group and one ordinate direction from the menus.
        `All angular directions` overlays every angular flux, while
        `Scalar flux only` hides them. With normalization enabled, angular and
        scalar fluxes are divided by the maximum scalar flux of the selected
        group, preserving their relative magnitudes.
    """),
    code(r"""
        INITIAL_GROUP = 0       # Zero-based Python group index
        INITIAL_DIRECTION = 'all'  # 'all', 'scalar', or a zero-based direction index
        NORMALIZE_FLUX = True

        flux_explorer = flux_widget(
            result,
            initial_group=INITIAL_GROUP,
            initial_direction=INITIAL_DIRECTION,
            normalize=NORMALIZE_FLUX,
        )
        display(flux_explorer)
    """),
]


def study_cells(orders: tuple[int, ...] = (2, 4, 6),
                target_mesh_widths: tuple[float, ...] = (4.0, 2.0, 1.0),
                run_by_default: bool = True,
                ) -> list[dict]:
    study_source = dedent(r"""
        # Extend these lists for a more detailed (and more expensive) study.
        ORDERS = __STUDY_ORDERS__                    # Even angular orders
        TARGET_MESH_WIDTHS = __STUDY_WIDTHS__  # Target cell widths in cm

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
    """).strip().replace('__STUDY_ORDERS__', repr(orders)).replace(
        '__STUDY_WIDTHS__', repr(target_mesh_widths)
    )
    guarded_study = (
        f"RUN_CONVERGENCE_STUDY = {run_by_default!r}\n\n"
        "if RUN_CONVERGENCE_STUDY:\n"
        f"{indent(study_source, '    ')}\n"
        "else:\n"
        "    print('Convergence study skipped; set RUN_CONVERGENCE_STUDY = True to run it.')"
    )
    return [
        markdown(r"""
        ## Angular and spatial convergence of $k_{eff}$

        Both panels report $(k-k_{\mathrm{finest}})\times10^5$ pcm, where
        $k_{\mathrm{finest}}$ is computed at the largest angular order and
        smallest target mesh width in the table below. Left: change $N$ for
        each fixed mesh. Right: change the spatial mesh for each fixed $N$.
        The number of energy groups and all material data stay fixed.
        Set `RUN_CONVERGENCE_STUDY = True` in the next cell to execute this
        optional, comparatively expensive study.
        """),
        code(guarded_study),
    ]


def critical_cells(title: str, description: str, case: str, multigroup: bool,
                   order: int, inner_tol: float, outer_tol: float,
                   eigen_tol: float, group_to_plot: int = 0,
                   target_mesh_width: float | None = None,
                   study_orders: tuple[int, ...] = (2, 4, 6),
                   study_widths: tuple[float, ...] = (4.0, 2.0, 1.0),
                   run_study_by_default: bool = True,
                   ) -> list[dict]:
    remesh_lines = ""
    if target_mesh_width is not None:
        remesh_lines = f"""
            problem = remesh_problem(solver.problem, {target_mesh_width!r})
            solver = SNSolver(problem, settings)
        """
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
            {remesh_lines}
            result = solver.solve_critical()

            print(f'k_eff = {{result.keff:.8f}}')
            print(f'power iterations = {{result.power_iterations}}')
            print(f'groups = {{result.scalar_flux.shape[0]}}')
            print(f'angular order = S_{{result.settings.order}}')
            print(f'spatial regions = {{result.problem.mesh.number_of_regions}}')
            print(f'spatial cells = {{result.problem.mesh.number_of_cells}}')
            print(f'materials = {{result.problem.cross_sections.region_materials}}')
        """),
        *ANALYSIS[:-1],
        code(f"""
            INITIAL_GROUP = {group_to_plot}  # Zero-based Python group index
            INITIAL_DIRECTION = 'all'  # 'all', 'scalar', or a direction index
            NORMALIZE_FLUX = True

            flux_explorer = flux_widget(
                result,
                initial_group=INITIAL_GROUP,
                initial_direction=INITIAL_DIRECTION,
                normalize=NORMALIZE_FLUX,
            )
            display(flux_explorer)
        """),
        *study_cells(study_orders, study_widths, run_study_by_default),
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
    ANALYSIS[4], ANALYSIS[5],
]


notebooks = {
    'Ex1_localized_isotropic_source.ipynb': source_cells,
    'Ex2_homogeneous_one_group.ipynb': critical_cells(
        'Ex2_homogeneous_one_group',
        'A homogeneous one group slab with vacuum boundary conditions.',
        'homogeneous', False, 4, 1e-6, 1e-4, 1e-6),
    'Ex3_homogeneous_multigroup.ipynb': critical_cells(
        'Ex3_three_region_multigroup',
        ('A reduced three-region reflector–fuel–reflector slab with four energy '
         'groups, energy coupling, and power iterations. The coarse teaching '
         'mesh keeps the interactive notebook fast.'),
        'reflected', True, 2, 1e-5, 1e-4, 1e-5, group_to_plot=3,
        target_mesh_width=4.0, study_orders=(2, 4), study_widths=(4.0, 2.0),
        run_study_by_default=False),
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
