"""Plotting functions that are generatlizable to both fooof and bycycle"""

import re
from scipy.stats import truncnorm
import plotly.graph_objects as go


def plot_scatter(param, label, urls):
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


def two_column_layout(fig_left, fig_right, graphs):
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

    Notes
    -----
    The css is defined in reports/templates/masthead.html.
    """

    graphs.append("<div class=plot-row>")
    graphs.append(re.sub("^<div>", "<div class=plot-column>", fig_left))
    graphs.append(re.sub("^<div>", "<div class=plot-column>", fig_right))
    graphs.append("</div>")

    return graphs
