"""Generate fooof/bycyle html reports."""

import os
from pathlib import Path
import codecs

from ndspflow.plts.fooof import plot_fooof_fit


SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>FOOOF Fits: {n_fooofs}</li>
\t\t<li>Bycycle Fits: {n_bycycles}</li>
\t</ul>
"""


def generate_report(fooof_graphs, bycycle_graphs, subject, out_dir,
                    fname, subject_template=SUBJECT_TEMPLATE):
    """Writes HTML reports for fooof fits to output directory.

    Parameters
    ----------
    fooof_graphs : list of str
        Fooof plots in the form of strings containing html generated from plotly.
    bycycle_graphs : list of str
        Bycycle plots in the form of strings containing html generated from plotly.
    subject : str
        Subject identifier.
    out_dir : str
        Directory to write the html page to.
    fname : str
        Name of the html file.
    subject_template : string, optional, default: SUBJECT_TEMPLATE
        Subject specific metadata to report.
    """

    # Load in template masthead
    cwd = Path(__file__).parent
    masthead = codecs.open(os.path.join(cwd, "templates/masthead.html"), "r", "utf-8")

    # Create the html report
    html_report = codecs.open(os.path.join(out_dir, "fooof.html"), "w+", "utf-8")

    # Extend the masthead template into the html report
    html_report.write(masthead.read())
    html_report.close()

    # Substitute html
    html_report = codecs.open(os.path.join(out_dir, "fooof.html"), "r", "utf-8")
    html_report_sub = html_report.read()

    # Subject info
    subject_template = subject_template.format(subject_id=subject, n_fooofs=len(fooof_graphs),
                                               n_bycycles=len(bycycle_graphs))
    html_report_sub = html_report_sub.replace("{% SUBJECT_TEMPLATE %}", subject_template)

    # FOOOF plots
    html_report_sub = html_report_sub.replace("{% fooof_graph %}", fooof_graphs)

    # Overwrite
    with open(os.path.join(out_dir, "fooof.html"), "r+") as html:
        html.read()
        html.seek(0)
        html.write(html_report_sub)
        html.truncate()

    # Close opened files
    html_report.close()
    masthead.close()

