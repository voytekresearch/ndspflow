"""Generate fooof/bycyle html reports."""

import os
from pathlib import Path

from fooof.core.strings import gen_settings_str, gen_results_fm_str


def generate_1d_report(fm, fooof_graph, subject,  n_fooofs, n_bycycles, out_dir, fname):
    """Generate reports for a single spectrum.

    Parameters
    ----------
    fm : fooof FOOOF
        A FOOOF object that has been fit.
    fooof_graph : list of str
        FOOOOF plot in the form of strings containing html generated from plotly.
    subject : str
        Subject identifier.
    n_fooofs : int
        The number of fooof fits.
    n_bycycles : int
        The number of bycycle fits.
    out_dir : str
        Directory to write the html page to.
    fname : str
        Name of the html file.
    """

    # Inject header and fooof report
    html_report = generate_header(subject, 1, 0)
    html_report = generate_fooof_report(fm, fooof_graph, html_report)

    # Write the html to a file
    with open(os.path.join(out_dir, fname), "w+") as html:
        html.write(html_report)


def generate_header(subject, n_fooofs, n_bycycles):
    """Include masthead and subject info in a HTML string.

    Parameters
    ----------
    subject : str
        Subject identifier.
    n_fooofs : int
        The number of fooof fits.
    n_bycycles : int
        The number of bycycle fits.
    subject_template : string, optional, default: SUBJECT_TEMPLATE
        Subject specific metadata to report.

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

    # Subject info
    subject_template = """\
    \t<ul class="elem-desc">
    \t\t<li>Spectrum ID: {subject_id}</li>
    \t\t<li>FOOOF Fits: {n_fooofs}</li>
    \t\t<li>Bycycle Fits: {n_bycycles}</li>
    \t</ul>
    """.format(subject_id=subject, n_fooofs=n_fooofs, n_bycycles=n_bycycles)

    html_report = html_report.replace("{% SUBJECT_TEMPLATE %}", subject_template)

    return html_report


def generate_fooof_report(fm, fooof_graph, html_report):
    """Include fooof settings, results, and plots in a HTML string.

    Parameters
    ----------
    fm : fooof FOOOF
        A FOOOF object that has been fit.
    fooof_graph : list of str
        FOOOOF plot in the form of strings containing html generated from plotly.
    html_report : str
        A string containing the html fooof report.

     Returns
    -------
    html_report : str
        A string containing the html fooof_report.
    """

    # Get html ready strings for settings and results
    settings = gen_settings_str(fm, False, True)
    results = gen_results_fm_str(fm, True)

    settings = settings.replace("\n","<br />\n")
    results = results.replace("\n","<br />\n")

    # FOOOF plots
    html_report = html_report.replace("{% fooof_settings %}", settings)
    html_report = html_report.replace("{% fooof_results %}", results)
    html_report = html_report.replace("{% fooof_graph %}", fooof_graph)

    return html_report

