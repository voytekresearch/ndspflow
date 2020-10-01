"""Functions to save results and plots."""

import os

from fooof import FOOOF, FOOOFGroup
from ndspflow.io.paths import clean_mkdir
from ndspflow.plts.fooof import plot_fooof
from ndspflow.reports.html import generate_1d_report


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
    dir_fooof = os.path.join(output_dir, 'fooof')
    clean_mkdir(dir_fooof)

    # Flatten the models and output dirs into a 1d list
    fms = []
    out_paths = []
    labels = []

    if type(model) is FOOOF:

        # For 1d arrays
        out_paths.append(dir_fooof)
        fms.append(model)
        labels.append("spectrum_single")

    elif type(model) is FOOOFGroup:

        # For 2d arrays
        label_template = "spectrum_dim1-{dim_a}"
        for fm_idx in range(len(model)):

            label = label_template.format(dim_a=str(fm_idx).zfill(4))
            labels.append(label)

            out_paths.append(os.path.join(dir_fooof, label))
            fms.append(model.get_fooof(fm_idx))

    elif type(model) is list:
        # For 3d arrays
        label_template = "spectrum_dim1-{dim_a}_dim2-{dim_b}"

        for fg_idx in range(len(model)):

            for fm_idx in range(len(model[0].get_results())):

                label = label_template.format(dim_a=str(fg_idx).zfill(4),
                                              dim_b=str(fm_idx).zfill(4))
                labels.append(label)
                out_paths.append(os.path.join(dir_fooof, label))
                fms.append(model[fg_idx].get_fooof(fm_idx))

    # Save outputs
    for fm, out_path, label in zip(fms, out_paths, labels):

        # Make the output directory
        clean_mkdir(out_path)

        # Save the model
        fm.save('results', file_path=out_path, append=False, save_results=True, save_settings=True)

        # Save the plot
        generate_1d_report(fm, plot_fooof(fm), label, 0, 0, out_path, 'report.html')
