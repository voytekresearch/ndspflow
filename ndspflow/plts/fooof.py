"""FOOOF plotting functions for returning ready-to-embed html."""

import re
import numpy as np
from scipy.stats import truncnorm
import plotly.graph_objects as go


def plot_fm(fm):
    """Plot a individual FOOOF fits using plotly.

    Parameters
    ----------
    fm : fooof FOOOF
        A fooof model that has been

    Returns
    -------
    graph : str
        The fooof plot as a string containing html.
    """

    # Create figure
    fig = go.Figure()

    config = {'responsive': True}

    # Original
    fig.add_trace(go.Scatter(x=fm.freqs, y=fm.power_spectrum, mode='lines',
                                name='Original Spectrum', line=dict(color='black', width=3)))

    # Model
    fig.add_trace(go.Scatter(x=fm.freqs, y=fm.fooofed_spectrum_, mode='lines',
                                name='Full Model Fit', line=dict(color='rgba(214, 39, 40, .7)',
                                                                width=3)))

    # Aperiodic
    fig.add_trace(go.Scatter(x=fm.freqs, y=fm._ap_fit, mode='lines', name='Aperiodic Fit',
                            line=dict(color='rgba(31, 119, 180, .7)', width=3, dash='dash')))

    # Plot settings
    fig.update_layout(
        title="Spectrum Fit",
        xaxis_title="Frequency",
        yaxis_title="Power",
        font=dict(
            family="sans-serif",
            size=18,
            color='black'
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    graph = fig.to_html(full_html=False, default_height='475', default_width='700',
                        include_plotlyjs=False)

    return graph


def plot_fgs(fgs, urls):
    """Plot a list of FOOOFGroup parameters distributions using plotly.

    Parameters
    ----------
    fgs : list of fooof FOOOFGroup
        FOOOFGroup objects that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    urls : list of str
        Local html paths to link points to their individual reports.

    Returns
    -------
    graphs : str
        Multiple FOOOFGroup plots as a string containing html.
    """

    graphs = []

    urls_reshape = np.reshape(urls, (len(fgs), int(len(urls)/len(fgs))))

    for idx, (fg, url) in enumerate(zip(fgs, urls_reshape)):

        #graphs.append("<br><br><h1 style=\"text-align:left\">Group Index: {idx}</h1>"\
        #    .format(idx=str(idx).zfill(4)))

        graphs.append(
            "<br><br><button type=\"button\" class=\"collapsible\">"\
            "<h1 style=\"text-align:left\">▼ Group Index: {idx}</h1></button>"\
            "<div class=\"content\">"\
            .format(idx=str(idx).zfill(4))
        )

        graphs.append(plot_fg(fg, url) + "</div>")

    return graphs


def plot_fg(fg, urls):
    """Plot FOOOFGroup parameters distributions using plotly.

    Parameters
    ----------
    fg : fooof FOOOFGroup
        FOOOFGroup object that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    urls : list of str
        Local html paths to link points to their individual reports.

    Returns
    -------
    graphs : str
        Multiple FOOOFGroup plots as a string containing html.
    """

    graphs = []

    # Fit
    graphs.append('<br><br><center><h1>Goodness of Fit</h1></center>')

    fig_error = plot_scatter(fg.get_params('error'), 'Error', urls)
    fig_rsq = plot_scatter(fg.get_params('r_squared'), 'R-Squared', urls)

    graphs = _two_column_layout(fig_error, fig_rsq, graphs)

    # PE params
    peak_idx = fg.get_params('peak_params', 'CF')[:, 1].astype(int)
    urls_peaks = [urls[idx] for idx in peak_idx]

    graphs.append('<br><br><center><h1>Peak Parameters</h1></center>')

    _, n_peaks = np.unique(peak_idx, return_counts=True)
    cfs = fg.get_params('peak_params', 'CF')[:, 0]

    fig_npeaks = plot_scatter(n_peaks, 'Number of Peaks', urls)
    fig_cf = plot_scatter(cfs, 'Center Frequency', urls_peaks)

    graphs = _two_column_layout(fig_npeaks, fig_cf, graphs)

    bws = fg.get_params('peak_params', 'BW')[:, 0]
    pws = fg.get_params('peak_params', 'PW')[:, 0]

    fig_bws = plot_scatter(bws, 'Band Width', urls_peaks)
    fig_pws = plot_scatter(bws, 'Peak Width', urls_peaks)

    graphs = _two_column_layout(fig_bws, fig_pws, graphs)

    # AP params
    graphs.append('<br><br><center><h1>Aperiodic Parameters</h1></center>')

    fig_exp = plot_scatter(fg.get_params('aperiodic', 'exponent'), 'Exponent', urls)
    fig_off = plot_scatter(fg.get_params('aperiodic', 'offset'), 'Offset', urls, yfmt=".2f")

    graphs = _two_column_layout(fig_exp, fig_off, graphs)

    # Place each item in the list on a newline
    graphs = "\n".join(graphs)

    return graphs



def plot_scatter(param, label, urls, yfmt=".3f"):
    """Plot an interactive scatterplot for a fit parameter.

    Parameters
    ----------
    param : 1d array
        A parameter from a FOOOF object that as been fit. Typically accessed using
        :meth:`fooof.FOOOFGroup.get_params`.
    label : str
        The label associated with the parameter that will be displayed on the plot's axes.
    urls : list of str
        Local html paths to link points to their individual reports.
    yfmt : str, optional, default: ".3f"
        The format to the y-axis labels used to ensure labels are all the same length, ensuring
        plots are aligned.


    Returns
    -------
    graph : str
        The fooof plot as a string containing html and embedded javascript.
    """

    # Generate random jitter for catplot
    lower, upper = 0.5, 1.5
    mu, sigma = 1, 0.05
    size = len(param)
    jitter = truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma,
                        loc=mu, scale=sigma,size=size)

    # Create scatterplot
    fig = go.Figure([go.Scatter(y=param, x=jitter, mode='markers', customdata=urls, hoverinfo='y')])

    # Plot settings
    fig.update_layout(
        xaxis = dict(title=label, showticklabels=False),
        yaxis = dict(title=label),
        font = dict(
            family="sans-serif",
            size=18,
            color='black'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        width=500,
        hovermode='closest'
    )
    fig.update_xaxes(range=[.5, 1.5])

    # Generate html
    graph = fig.to_html(full_html=False, default_height='475', default_width='700',
                        include_plotlyjs=False)

    # Get the html div id associated with the plot
    res = re.search('<div id="([^"]*)"', graph)
    div_id = res.groups()[0]

    # Build javaScript callback linking point clicks
    js_callback = """
        <script>
        var plot_element = document.getElementById("{div_id}");
        plot_element.on('plotly_click', function(data){{
            console.log(data);
            var point = data.points[0];
            var event = data || window.event;
            if (point) {{
                console.log(point.customdata);
                if (event.ctrlKey || event.metaKey) {{
                    window.open(point.customdata,"_self");
                }} else {{
                    window.location = point.customdata;
                }}

            }}
        }})
        </script>
        """.format(div_id=div_id)

    graph = graph + '\n' + js_callback
    return graph


def _two_column_layout(fig_left, fig_right, graphs):
    """Place two figure on the same row using html and css.

    Parameters
    ----------
    fig_left : str
        Html code from a plotly fig, using the to_html method, to place in the left column.
    fig_left : str
        Html code from a plotly fig, using the to_html method, to place in the left column.
    graphs : list of str
       Contains html plot strings.

    Returns
    -------
    graphs : list of str
        Html plot strings with the double column plots appended.

    Note
    ----
    The css is defined in reports/templates/masthead.html.
    """

    graphs.append("<div class=plot-row>")
    graphs.append(re.sub("^<div>", "<div class=plot-column>", fig_left))
    graphs.append(re.sub("^<div>", "<div class=plot-column>", fig_right))
    graphs.append("</div>")

    return graphs
