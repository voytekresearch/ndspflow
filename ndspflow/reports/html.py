"""Generate fooof/bycyle html reports."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from fooof import FOOOF, FOOOFGroup
from fooof.core.strings import gen_results_fm_str, gen_results_fg_str
from fooof.core.strings import gen_settings_str as gen_settings_fm_str

from ndspflow.core.utils import flatten_fms, flatten_bms
from ndspflow.plts.fooof import plot_fm, plot_fg, plot_fgs
from ndspflow.plts.bycycle import plot_bm, plot_bg, plot_bgs


def generate_report(output_dir, fms=None, bms=None, group_fname='report_group.html'):
    """Generate all FOOOF and/or Bycycle reports.

    Parameters
    ----------
    output_dir : str
        The path to write the reports to.
    fms : fooof FOOOF, FOOOFGroup, or list of FOOOFGroup, optional, default: None
        FOOOF object(s) that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    bms : tuple of (pd.DataFrame or list of pd.DataFrame, dict)
        Bycycle dataframes organized as (df_features, bycycle_args), where df_features is returned
        using bycycle_args as arg and kwargs, from func:`~.fit_bycycle`.
    group_fname : str, optional, default: 'report_group.html'
        The name of the group report file.
    """

    # Infer number of bycycle models
    n_bms = 0 if bms is None else None
    n_bms = 1 if n_bms is None and isinstance(bms[0], pd.DataFrame) else n_bms
    n_bms = len(bms[0]) if n_bms is None and isinstance(bms[0][0], pd.DataFrame) else n_bms
    n_bms = int(np.prod(np.shape(bms)[:2])) if n_bms is None and \
        isinstance(bms[0][0][0], pd.DataFrame) else n_bms

    # Infer number of fooof models
    n_fms = 0 if fms is None else None
    n_fms = 1 if isinstance(fms, FOOOF) else n_fms
    n_fms = len(fms) if isinstance(fms, FOOOFGroup) else n_fms
    n_fms = int(len(fms) * len(fms[0])) if isinstance(fms, list) else n_fms

    # Navigation page
    nav_page = []
    if fms is not None:

        # Generate fooof reports
        nav_page.append("<h1>FOOOF Reports</h1>""")

        fooof_dir = os.path.join(output_dir, "fooof")
        fm_list, fm_paths = flatten_fms(fms, fooof_dir)

        group_url = str('file://' + os.path.join(fooof_dir, group_fname)) if n_fms > 0 else None

        urls =  [str('file://' + os.path.join(fm_path, "report.html")) for fm_path in fm_paths]

        # Individual spectrum reports
        nav_links = []
        for fm, fm_path in zip(fm_list, fm_paths):

            # Inject header and fooof report
            label = fm_path.split('/')[-1]
            html_report = generate_header("subject", output_dir, "fooof", n_fooofs=n_fms,
                                          n_bycycles=n_bms, label=label, group_link=group_url)

            graph = plot_fm(fm).to_html(full_html=False, default_height='475',
                                        default_width='700', include_plotlyjs=False)

            html_report = generate_fooof_report(fm, graph, html_report)

            # Write the html to a file
            url = os.path.join(fm_path, 'report.html')
            with open(url, "w+") as html:
                html.write(html_report)

            # For the navigation page
            nav_links.append("<h3><a href=\"{url}\">{label}</a></h3>".format(url=url, label=label))

        # Group spectra reports
        if type(fms) is FOOOFGroup:

            # Inject header and fooof report
            html_report = generate_header("group", output_dir, "fooof", n_fooofs=n_fms,
                                          n_bycycles=n_bms, group_link=group_url)

            html_report = generate_fooof_report(fms, plot_fg(fms, urls), html_report)

        elif type(fms) is list:

            # Inject header
            html_report = generate_header("group", output_dir, "fooof", n_fooofs=n_fms,
                                          n_bycycles=n_bms, group_link=group_url)

            html_report = generate_fooof_report(fms, plot_fgs(fms, urls), html_report)

        if type(fms) is FOOOFGroup or type(fms) is list:

            # Write the html to a file
            group_url = os.path.join(fooof_dir, group_fname)
            with open(group_url, "w+") as html:
                html.write(html_report)

            group_link = "<h3><a href=\"{url}\">Group</a></h3>".format(url=group_url)
            nav_page.append(group_link)

        nav_page.append("\n".join(nav_links))

        # Create a fooof navigation page
        html_report = generate_header("nav", output_dir, "fooof", n_fooofs=n_fms,
                                       n_bycycles=n_bms, group_link=group_url)

        html_report = html_report.replace("{% NAV %}", "\n".join(nav_page))

        with open(os.path.join(fooof_dir, "nav.html"), "w") as html:
            html.write(html_report)

    nav_page = []
    if bms is not None:

        # Unpack tuple
        (dfs_features, fit_kwargs) = bms

        sigs = fit_kwargs.pop('sig')
        fs = fit_kwargs.pop('fs')
        thresholds = fit_kwargs['threshold_kwargs']

        # Generate bycycle reports
        nav_page.append("<h1>Bycycle Reports</h1>")

        bycycle_dir = os.path.join(output_dir, "bycycle")

        dfs_features_2d, bc_paths, sigs_2d = flatten_bms(dfs_features, bycycle_dir, sigs=sigs)

        group_url = str('file://' + os.path.join(bycycle_dir, group_fname)) if n_bms > 1 else None

        # Individual signal reports
        nav_links = []
        for idx, (df_features, sig, bc_path) in enumerate(zip(dfs_features_2d, sigs_2d, bc_paths)):

            # Inject header and bycycle report
            label = bc_path.split('/')[-1]
            html_report = generate_header("subject", output_dir, "bycycle", n_fooofs=n_fms,
                                          n_bycycles=n_bms, label=label, group_link=group_url)

            graph = plot_bm(df_features, sig, fs, thresholds, idx, plot_only_result=False)

            html_report = generate_bycycle_report(fit_kwargs, graph, html_report)

            # Write the html to a file
            url = os.path.join(bc_path, 'report.html')
            with open(url, "w+") as html:
                html.write(html_report)

            # For the navigation page
            nav_links.append("<h3><a href=\"{url}\">{label}</a></h3>".format(url=url, label=label))

        # Mutlidim reports
        if sigs.ndim == 2:

            graph = plot_bg(dfs_features, sigs, fs)

        elif sigs.ndim == 3:

            graph = plot_bgs(dfs_features, sigs, fs)

        if sigs.ndim == 2 or sigs.ndim == 3:

            html_report = generate_header("group", output_dir, "bycycle", n_fooofs=n_fms,
                                          n_bycycles=n_bms, group_link=group_url)

            html_report = generate_bycycle_report(fit_kwargs, graph, html_report)

            # Write the html to a file
            group_url = os.path.join(bycycle_dir, group_fname)
            with open(group_url, "w+") as html:
                html.write(html_report)

            group_link = "<h3><a href=\"{url}\">Group</a></h3>".format(url=group_url)
            nav_page.append(group_link)

        nav_page.append("\n".join(nav_links))

        # Create a bycycle navigation page
        html_report = generate_header("nav", output_dir, "bycycle", n_fooofs=n_fms,
                                      n_bycycles=n_bms, group_link=group_url)

        html_report = html_report.replace("{% NAV %}", "\n".join(nav_page))

        with open(os.path.join(bycycle_dir, "nav.html"), "w") as html:
            html.write(html_report)


def generate_header(report_type, output_dir, dtype, label=None,
                    n_fooofs=0, n_bycycles=0, group_link=None):
    """Include masthead and subject info in a HTML string.

    Parameters
    ----------
    report_type : str, {'subject', 'group'}
        Specifices header metadata for 1d arrays versus 2d/3d arrays.
    dtype : str, {'fooof', 'bycycle'}
        Specifices header metadata for fooof versus bycycle reports.
    output_dir : str, optional, default: None
        The path to write the reports to.
    label : list of str, optional, default: None
        Spectrum identifier.
    n_fooofs : int, optional, default: 0
        The number of fooof fits.
    n_bycycles : int, optional, default: 0
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
        \t\t<li>{dtype} ID: {label}</li>
        \t</ul>
        """.format(dtype=dtype, label=label)

    elif report_type == "group":
        # Read in body
        body = open(os.path.join(cwd, "templates/group.html"), "r")
        body_template = body.read()
        body.close()

    elif report_type == "nav":
        # Read in body
        body = open(os.path.join(cwd, "templates/nav.html"), "r")
        body_template = body.read()
        body.close()

    if report_type in ["group", "nav"]:
        # Set meta string
        meta_template = """
        \t<ul class="elem-desc">
        \t\t<li>Group Report</li>
        \t\t<li>FOOOF Fits: {n_fooofs}</li>
        \t\t<li>Bycycle Fits: {n_bycycles}</li>
        \t</ul>
        """.format(n_fooofs=n_fooofs, n_bycycles=n_bycycles)

    group_link = "" if group_link is None else group_link
    root_path = os.path.join(cwd, "templates")
    html_report = html_report.replace("{% GROUP %}", group_link)
    html_report = html_report.replace("{% BODY %}", body_template)
    html_report = html_report.replace("{% META_TEMPLATE %}", meta_template)
    html_report = html_report.replace("{% ROOT %}", root_path)

    # Navbar
    fooof_nav = os.path.join(output_dir, 'fooof', "nav.html")
    bycycle_nav = os.path.join(output_dir, 'bycycle', "nav.html")
    fooof_str = "FOOOF" if n_fooofs > 0 else ""
    bycycle_str = "Bycycle" if n_bycycles > 0 else ""

    html_report = html_report.replace("{% NAV_FOOOF %}", fooof_nav)
    html_report = html_report.replace("{% NAV_BYCYCLE %}", bycycle_nav)
    html_report = html_report.replace("{% FOOOF %}", fooof_str)
    html_report = html_report.replace("{% Bycycle %}", bycycle_str)

    if dtype == "bycycle":
        html_report = html_report.replace("{% OUT %}", output_dir)
    else:
        html_report = html_report.replace("{% OUT %}", "")

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

        settings = gen_settings_fm_str(model, False, True)
        results = gen_results_fm_str(model, True)

    elif type(model) is FOOOFGroup:

        settings = gen_settings_fm_str(model, False, True)
        results = gen_results_fg_str(model, True)

    elif type(model) is list:

        settings = gen_settings_fm_str(model[0], False, True)
        results = [gen_results_fg_str(fg, True) for fg in model]

    # String formatting
    if type(model) is list:

        # Merge results and graphs together
        results = [result.replace("\n", "<br />\n") for result in results]

        for idx, graph in enumerate(fooof_graphs[::2].copy()):
            fooof_graphs.insert(fooof_graphs.index(graph) + 1, results[idx])

        results = "<br />\n".join(fooof_graphs)

        # Results now contain graphs, so drop liquid variable for graphs
        html_report = html_report.replace("{% graph %}", "")

    else:
        results = results.replace("\n", "<br />\n")
        html_report = html_report.replace("{% graph %}", fooof_graphs)

    settings = settings.replace("\n", "<br />\n")

    # Inject settings and results
    html_report = html_report.replace("{% model_type %}", 'FOOOF')
    html_report = html_report.replace("{% settings %}", settings)
    html_report = html_report.replace("{% results %}", results)

    return html_report


def generate_bycycle_report(fit_kwargs, graph, html_report):
    """Include bycycle settings, results, and plots in a HTML string.

    Parameters
    ----------
    fit_kwargs : dict
        All args and kwargs used in :func:`~.fit_bycycle`.
    graph : str
        Contains plotly html and javascript.
    html_report : str
        A string containing the html bycycle report.

    Returns
    -------
    html_report : str
        A string containing the html bycycle report.
    """

    # Create a settings string
    fit_params = ['Frequency Range', 'Center Extrema', 'Burst Method']
    thr_params = ['Amplitude Fraction Threshold', 'Amplitude Consistency Threshold',
                  'Period Consistency Threshold', 'Monotonicity Threshold',
                  'Minimum Consecutive Cycles']

    fit_str = ["{key}: {value}".format(key=key.replace("_", " ").title(), value=value)
               for key, value in zip(fit_params, list(fit_kwargs.values())[:-2])]
    thr_str = ["{key}: {value}".format(key=key.replace("_", " ").title(), value=value)
               for key, value in zip(thr_params, fit_kwargs['threshold_kwargs'].values())]

    settings = [
        "=",
        'BYCYCLE - SETTINGS',
        *fit_str,
        *thr_str,
        "="
    ]
    settings[0] = settings[0] * 70
    settings[-1] = settings[-1] * 70
    settings = "<br />\n".join([line.center(70) for line in settings])

    # Inject settings and graph strings into template
    html_report = html_report.replace("{% model_type %}", 'Bycycle')
    html_report = html_report.replace("{% settings %}", settings)
    html_report = html_report.replace("{% graph %}", graph)
    html_report = html_report.replace("{% results %}", "")

    return html_report
