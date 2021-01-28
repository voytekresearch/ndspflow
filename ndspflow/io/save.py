"""Functions to save results and plots."""

import os
from pathlib import Path

from ndspflow.core.utils import flatten_fms, flatten_bms
from ndspflow.io.paths import clean_mkdir


def save_fooof(model, output_dir):
    """Make output directories and save FOOOF fits.

    Parameters
    ----------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit using :func:`ndspflow.core.fit.fit_fooof`.
    output_dir : str
        Path to write FOOOF results to.
    """

    # Make the fooof output dir
    fooof_dir = os.path.join(output_dir, 'fooof')
    clean_mkdir(fooof_dir)

    # Flatten model(s) and create output paths and sub-dir labels
    fms, out_paths = flatten_fms(model, fooof_dir)

    # Save outputs
    for fm, out_path in zip(fms, out_paths):

        # Make the output directory
        clean_mkdir(out_path)

        # Save the model
        fm.save('results', file_path=out_path, append=False, save_results=True, save_settings=True)


def save_bycycle(dfs_features, output_dir):
    """Make output directories and save bycycle dataframes.

    Parameters
    ----------
    dfs_features : pandas.DataFrame or list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
    output_dir : str
        Path to write FOOOF results to.
    """

    # Make the bycycle output dir
    bycycle_dir = os.path.join(output_dir, 'bycycle')
    clean_mkdir(bycycle_dir)

    dfs_features, bc_paths, _ = flatten_bms(dfs_features, bycycle_dir)

    # Save outputs
    for df_features, bc_path in zip(dfs_features, bc_paths):

        # Make the output directory
        clean_mkdir(bc_path)

        # Save the dataframes
        df_features.to_csv(os.path.join(bc_path, 'results.csv'), index=False)

        # Save javascript
        _save_bycycle_js(dfs_features, bycycle_dir)


def _save_bycycle_js(dfs_features, bycycle_dir):
    """Write out all required javascript."""

    # Convert dataframe to js array
    dfs_js = []

    for df_features in dfs_features:

        df_js = df_features.astype("str")
        df_js = df_js.to_numpy().tolist()
        df_js.insert(0, df_features.columns.tolist())
        dfs_js.append(df_js)

    # Get functions defined in custom.js
    cwd = Path(__file__).parent.parent
    with open(os.path.join(cwd, 'reports/templates/resources/js/custom.js'), 'r') as f:
        custom_js = f.read()

    # Create a function that fetches the array
    js_str = """
    function fetchData(index){{
        var dataFrame = {dfs_js};
        if (index == "None"){{
            return dataFrame;
        }}
        else {{
            return dataFrame[index];
        }};
    }}
    """.format(dfs_js=dfs_js)

    # Write out
    with open(os.path.join(bycycle_dir, '.data.js'), "w") as f:

        f.write("\n".join([js_str, custom_js]))
