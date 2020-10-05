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


def plot_fg(fg, urls):
    """Plot FOOOFGroup parameters distributions using plotly.

    Parameters
    ----------
    fg : fooof FOOOFGroup
        FOOOFGroup object that have been fit using :func:`ndspflow.core.fit.fit_fooof`.
    """

    cf_params = fg.get_params('peak_params', 'CF')
    cfs = np.array([param for idx, param in enumerate(cf_params[:, 0]) if idx % 2 == 0])
    # NEEDS FIX, MAPPING MULTIPLE CFS DOESNT WORK

    graphs = []
    graphs.append(plot_scatter(fg.get_params('error'), 'Error', urls))

    graphs = "\n".join(graphs)

    return graphs


def plot_scatter(param, label, urls):
    """Plot an interactive scatterplot for a fit parameter.

    Parameters
    ----------
    param : 1d array


    Returns
    -------
    graph : str
        The fooof plot as a string containing html.
    """

    # Generate random jitter for catplot
    lower, upper = 0.5, 1.5
    mu, sigma = 1, 0.05
    size = len(param)
    jitter = truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma,
                        loc=mu, scale=sigma,size=size)

    # Crate figure
    fig = go.Figure([go.Scatter(y=param, x=jitter, mode='markers', customdata=urls)])

    fig.update_layout(
        xaxis = dict(title=label, showticklabels=False),
        yaxis_title=label,
        font=dict(
            family="sans-serif",
            size=18,
            color='black'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        width=500,
        hovermode='closest'
    )
    fig.update_xaxes(range=[.5, 1.5])

    graph = fig.to_html(full_html=False, default_height='475', default_width='700',
                        include_plotlyjs=False)

    # Add a callback
    # Get id of html div element that looks like
    res = re.search('<div id="([^"]*)"', graph)
    div_id = res.groups()[0]

    # Build JavaScript callback for handling clicks
    # and opening the URL in the trace's customdata
    js_callback = """
        <script>
        var plot_element = document.getElementById("{div_id}");
        plot_element.on('plotly_click', function(data){{
            console.log(data);
            var point = data.points[0];
            if (point) {{
                console.log(point.customdata);
                var win = window.open(point.customdata, '_blank');
            }}
        }})
        </script>
        """.format(div_id=div_id)

    graph = graph + '\n' + js_callback

    return graph

