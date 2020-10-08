"""Generate fooof/bycyle html reports."""

import os
from pathlib import Path

from fooof import FOOOF, FOOOFGroup
from fooof.core.strings import gen_settings_str, gen_results_fm_str, gen_results_fg_str

from ndspflow.core.fit import flatten_fms
from ndspflow.plts.fooof import plot_fm, plot_fg, plot_fgs


def generate_report(output_dir, fms=None, bms=None, group_fname='report_group.html'):
    """Generate all FOOOF and/or Bycycle reports.

    Parameters
    ----------
    output_dir : str
        The path to write the reports to.
    fms : fooof FOOOF, FOOOFGroup, or list of FOOOFGroup, optional, default: None
        FOOOF object(s) that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    bms : bycycle objects
        Bycycle object(s) that have been fit.
    """

    # Generate fooof reports
    if fms is not None:

        fooof_dir = os.path.join(output_dir, 'fooof')
        fm_list, fm_paths, fm_labels = flatten_fms(fms, fooof_dir)

        group_url = str('file://' + os.path.join(fooof_dir, group_fname)) \
            if len(fm_list) > 1 else None

        for fm, fm_path, fm_label in zip(fm_list, fm_paths, fm_labels):

            generate_1d_report(fm, fm_label, plot_fm(fm), fm_path,
                               fname='report.html', group_link=group_url)

        urls =  [str('file://' + os.path.join(fm_path, 'report.html')) for fm_path in fm_paths]

        if type(fms) is FOOOFGroup:

            generate_2d_report(fms, plot_fg(fms, urls), len(fm_list), 0,
                               fooof_dir, fname=group_fname)

        elif type(fms) is list:

            generate_3d_report(fms, plot_fgs(fms, urls), int(len(fms)*len(fms[0])), 0,
                               fooof_dir, fname=group_fname)


def generate_1d_report(fm, fm_label, fooof_graph, out_dir, fname='report.html', group_link=None):
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
    html_report = generate_header('subject',fm_label=fm_label, group_link=group_link)
    html_report = generate_fooof_report(fm, fooof_graph, html_report)

    # Write the html to a file
    with open(os.path.join(out_dir, fname), "w+") as html:
        html.write(html_report)


def generate_2d_report(fg, fooof_graph, n_fooofs, n_bycycles, out_dir, fname='report_group.html'):
    """ Generate a group report for 2d arrays.

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
    group_link = str('file://' + os.path.join(out_dir, fname))
    html_report = generate_header('group', n_fooofs=len(fg), n_bycycles=0, group_link=group_link)
    html_report = generate_fooof_report(fg, fooof_graph, html_report)

    # Write the html to a file
    with open(os.path.join(out_dir, fname), "w+") as html:
        html.write(html_report)


def generate_3d_report(fgs, fooof_graph, n_fooofs, n_bycycles, out_dir, fname='report_group.html'):
    """ Generate a group report for 3d arrays.

    Parameters
    ----------
    fgs : list of fooof FOOOFGroup
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
    group_link = str('file://' + os.path.join(out_dir, fname))
    html_report = generate_header('group', n_fooofs=int(len(fgs)*len(fgs[0])),
                                  n_bycycles=0, group_link=group_link)
    html_report = generate_fooof_report(fgs, fooof_graph, html_report)

    # Write the html to a file
    with open(os.path.join(out_dir, fname), "w+") as html:
        html.write(html_report)


def generate_header(report_type, fm_label=None, n_fooofs=None, n_bycycles=None, group_link=None):
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
        \t\t<li>Individual Report</li>
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

    group_link = "" if group_link is None else group_link
    html_report = html_report.replace("{% GROUP %}", group_link)
    html_report = html_report.replace("{% BODY %}", body_template)
    html_report = html_report.replace("{% META_TEMPLATE %}", meta_template)

    return html_report


def generate_fooof_report(model, fooof_graphs, html_report):
    """Include fooof settings, results, and plots in a HTML string.

    Parameters
    ----------
    model : FOOOF, FOOOFGroup, or list of FOOOFGroup objects.
        A FOOOF object that has been fit using :func:`ndspflow.core.fit.fit_fooof`.
    fooof_graphs : 2d list of str
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

    elif type(model) is list:

        settings = gen_settings_str(model[0], False, True)
        results = [gen_results_fg_str(fg, True) for fg in model]

    # String formatting
    if type(model) is list:

        # Merge results and graphs together
        results = [result.replace("\n", "<br />\n") for result in results]

        for idx, graph in enumerate(fooof_graphs[::2].copy()):
            fooof_graphs.insert(fooof_graphs.index(graph) + 1, results[idx])

        results = "<br />\n".join(fooof_graphs)

        # Results now contain graphs, so drop liquid variable for graphs
        html_report = html_report.replace("{% fooof_graph %}", "")

    else:
        results = results.replace("\n", "<br />\n")
        html_report = html_report.replace("{% fooof_graph %}", fooof_graphs)


    settings = settings.replace("\n", "<br />\n")

    # Inject settings and results
    html_report = html_report.replace("{% fooof_settings %}", settings)
    html_report = html_report.replace("{% fooof_results %}", results)

    return html_report
