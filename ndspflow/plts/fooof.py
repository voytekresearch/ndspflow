"""FOOOF plotting functions for returning ready-to-embed html."""

import numpy as np
import plotly.graph_objects as go


def plot_fooof(fm):
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

    graph = fig.to_html(full_html=False, default_height='475', default_width='700')

    return graph
