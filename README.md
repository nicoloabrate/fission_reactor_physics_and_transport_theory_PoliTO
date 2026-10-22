# Reactor Physics and Transport Theory — PoliTO

Interactive exercises for the *Reactor Physics and Transport Theory* course at Politecnico di Torino.

## Run in the browser

No local Python or Jupyter installation is required. Open the exercise in JupyterLab:

**[Launch the multigroup-collapsing notebook](https://nicoloabrate.github.io/fission_reactor_physics_and_transport_theory_PoliTO/2026/lab/index.html?path=Ex_multigroup_collapsing.ipynb)**

The first launch can take a little longer while the browser downloads the Python runtime. Changes made by each student are stored only in that browser. To restore the published version, clear the JupyterLite site data from the browser or open the site in a private window.

## Deployment

The site is built with [JupyterLite](https://jupyterlite.readthedocs.io/) and deployed automatically to GitHub Pages after every push to `main`.

Repository settings required on GitHub:

1. Open **Settings → Pages** and select **GitHub Actions** as the source.
2. Open **Actions** and enable workflows if they are disabled.
3. Push to `main`, then wait for the **Build and Deploy** workflow to finish.

The published site is available at:

<https://nicoloabrate.github.io/fission_reactor_physics_and_transport_theory_PoliTO/>

## Local preview

Create a Python environment, then run:

```bash
python -m pip install -r requirements.txt
jupyter lite serve
```

The exercise directory is copied into the browser filesystem at build time. `MGC.py` deliberately keeps its JSON workflow free of server-only dependencies so it can run under the Pyodide kernel.
