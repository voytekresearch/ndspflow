"""Generate fooof/bycyle html reports."""

import os
from pathlib import Path

from fooof import FOOOF, FOOOFGroup
from fooof.core.strings import gen_settings_str, gen_results_fm_str, gen_results_fg_str

from ndspflow.core.fit import flatten_fms
from ndspflow.plts.fooof import plot_fm, plot_fg


def generate_report(output_dir, fms=None, bms=None):
    """


    fm_labels : list of str
        Spectrum identifiers.
    """

    # Generate fooof reports
    if fms is not None:

        fooof_dir = os.path.join(output_dir, 'fooof')
        fm_list, fm_paths, fm_labels = flatten_fms(fms, fooof_dir)

        for fm, fm_path, fm_label in zip(fm_list, fm_paths, fm_labels):

            generate_1d_report(fm, fm_label, plot_fm(fm), fm_path, 'report.html')

        if len(fm_list) > 1:

            urls =  [str('file://' + os.path.join(fm_path, 'report.html')) for fm_path, fm_label in zip(fm_paths, fm_labels)]
            graph = plot_fg(fms, urls)

            generate_2d_report(fms, graph, len(fm_list), 0, fooof_dir)


def generate_1d_report(fm, fm_label, fooof_graph, out_dir, fname='report.html'):
    """Generate reports for a single spectrum.

    Parameters
    ----------
    fm : fooof FOOOF
        A FOOOF object that has been fit.
    fm_label : list of str, optional, default: None
        Spectrum identifier.
    fooof_graph : list of str
        FOOOOF plot in the form of strings containing html generated from plotly.
    out_dir : str
        Directory to write the html page to.
    fname : str, optional, default: 'report.html'
        Name of the html file.
    """

    # Inject header and fooof report
    html_report = generate_header('subject',fm_label=fm_label)
    html_report = generate_fooof_report(fm, fooof_graph, html_report)

    # Write the html to a file
    with open(os.path.join(out_dir, fname), "w+") as html:
        html.write(html_report)


def generate_2d_report(fg, fooof_graph, n_fooofs, n_bycycles, out_dir, fname='report_group.html'):
    """ Generate group report for a 2d array input.

    Parameters
    ----------
    fg : fooof FOOOFGroup
        FOOOFGroup object that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    fooof_graph : list of str
        FOOOOF plot in the form of strings containing html generated from plotly.
    n_fooofs : int
        The number of fooof fits.
    n_bycycles : int
        The number of bycycle fits.
    out_dir : str
        Directory to write the html page to.
    fname : str, optional, default: 'report_group.html'
        Name of the html file.
    """

    # Inject header
    html_report = generate_header('group', n_fooofs=len(fg), n_bycycles=0)
    html_report = generate_fooof_report(fg, fooof_graph, html_report)

    # Write the html to a file
    with open(os.path.join(out_dir, fname), "w+") as html:
        html.write(html_report)


def generate_header(report_type, fm_label=None, n_fooofs=None, n_bycycles=None):
    """Include masthead and subject info in a HTML string.

    Parameters
    ----------
    report_type : str, {subject, group}
        Specifices header metadata for 1d arrays versus 2d/3d arrays.
    fm_label : list of str, optional, default: None
        Spectrum identifier.
    n_fooofs : int, optional, default: None
        The number of fooof fits.
    n_bycycles : int, optional, default: None
        The number of bycycle fits.

    Returns
    -------
    html_report : str
        A string containing the html header to insert into the fooof report.
    """

    # Load in template masthead
    cwd = Path(__file__).parent

    masthead = open(os.path.join(cwd, "templates/masthead.html"), "r")
    html_report = masthead.read()
    masthead.close()

    # Report metadata
    if report_type == "subject":
        # Read in body
        body = open(os.path.join(cwd, "templates/subject.html"), "r")
        body_template = body.read()
        body.close()

        # Set meta string
        meta_template = """\
        \t<ul class="elem-desc">
        \t\t<li>Subject Report</li>
        \t\t<li>Spectrum ID: {spectrum_id}</li>
        \t</ul>
        """.format(spectrum_id=fm_label)

    if report_type == "group":
        # Read in body
        body = open(os.path.join(cwd, "templates/group.html"), "r")
        body_template = body.read()
        body.close()

        # Set meta string
        meta_template = """\
        \t<ul class="elem-desc">
        \t\t<li>Group Report</li>
        \t\t<li>FOOOF Fits: {n_fooofs}</li>
        \t\t<li>Bycycle Fits: {n_bycycles}</li>
        \t</ul>
        """.format(n_fooofs=n_fooofs, n_bycycles=n_bycycles)

    html_report = html_report.replace("{% BODY %}", body_template)
    html_report = html_report.replace("{% META_TEMPLATE %}", meta_template)

    return html_report


def generate_fooof_report(model, fooof_graph, html_report):
    """Include fooof settings, results, and plots in a HTML string.

    Parameters
    ----------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit using :func:`ndspflow.core.fit.fit_fooof`.
    fooof_graph : list of str
        FOOOOF plot in the form of strings containing html generated from plotly.
    html_report : str
        A string containing the html fooof report.

     Returns
    -------
    html_report : str
        A string containing the html fooof_report.
    """

    # Get html-ready strings for settings and results
    if type(model) is FOOOF:

        settings = gen_settings_str(model, False, True)
        results = gen_results_fm_str(model, True)

    elif type(model) is FOOOFGroup:

        settings = gen_settings_str(model, False, True)
        results = gen_results_fg_str(model, True)

    settings = settings.replace("\n","<br />\n")
    results = results.replace("\n","<br />\n")

    # FOOOF plots
    html_report = html_report.replace("{% fooof_settings %}", settings)
    html_report = html_report.replace("{% fooof_results %}", results)
    html_report = html_report.replace("{% fooof_graph %}", fooof_graph)

    return html_report
