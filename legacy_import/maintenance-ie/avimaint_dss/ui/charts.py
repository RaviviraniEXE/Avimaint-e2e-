"""Plotly charts styled with the validated palette. Single-axis, accessible."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .theme import HAIR, INK, INK2, MUTED, PLANE, SERIES, STATUS, SURFACE

_FONT = "system-ui,-apple-system,Segoe UI,sans-serif"


def _base(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=INK2, size=13),
        margin=dict(l=8, r=14, t=10, b=8), showlegend=False,
        xaxis=dict(gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR),
        yaxis=dict(gridcolor=HAIR, zerolinecolor=HAIR, linecolor=HAIR),
        hoverlabel=dict(bgcolor=SURFACE, font_size=12, font_family=_FONT, bordercolor=HAIR),
    )
    return fig


def hbar(df: pd.DataFrame, cat: str, val: str, color: str = SERIES[0],
         height: int = 360, suffix: str = "") -> go.Figure:
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[val], y=d[cat], orientation="h", marker=dict(color=color, line=dict(width=0)),
        text=[f"{v:,}{suffix}" for v in d[val]], textposition="outside",
        textfont=dict(color=INK2, size=12),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    fig = _base(fig, height)
    fig.update_xaxes(showgrid=True); fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=8, r=48, t=8, b=8))
    return fig


def pareto(df: pd.DataFrame, cat: str, share_col: str, cum_col: str,
           height: int = 380) -> go.Figure:
    """Single 0-100 axis: share% bars + cumulative% line (no dual axis)."""
    fig = go.Figure()
    fig.add_bar(x=df[cat], y=df[share_col], marker=dict(color=SERIES[0]),
                name="Share %", hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=df[cat], y=df[cum_col], mode="lines+markers",
                    line=dict(color=SERIES[1], width=2), marker=dict(size=7),
                    name="Cumulative %", hovertemplate="%{x}: %{y:.1f}% cumulative<extra></extra>")
    fig = _base(fig, height)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0,
                      bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)))
    fig.update_yaxes(title="% of work orders", range=[0, 105])
    fig.update_xaxes(tickangle=-35)
    return fig


def heatmap(mat: pd.DataFrame, height: int = 420) -> go.Figure:
    # sequential blue ramp
    colorscale = [[0.0, "#eff6ff"], [0.18, "#bfdbfe"], [0.55, "#60a5fa"], [1.0, "#2563eb"]]
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index),
        colorscale=colorscale, showscale=True,
        colorbar=dict(outlinecolor=HAIR, tickfont=dict(color=MUTED)),
        hovertemplate="%{y} · %{x}: %{z} work orders<extra></extra>",
        text=mat.values, texttemplate="%{text}", textfont=dict(size=11, color=INK2),
    ))
    fig = _base(fig, height)
    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def action_bars(df: pd.DataFrame, cat: str, val: str, height: int = 320) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df[cat], y=df[val], marker=dict(color=SERIES[2]),
        text=df[val], textposition="outside", textfont=dict(color=INK2, size=12),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig = _base(fig, height)
    fig.update_xaxes(showgrid=False)
    return fig


def outcome_donut(df: pd.DataFrame, height: int = 300) -> go.Figure:
    cmap = {"positive": STATUS["good"], "unknown": MUTED,
            "negative": STATUS["critical"], "mixed": STATUS["serious"]}
    colors = [cmap.get(o, MUTED) for o in df["outcome"]]
    fig = go.Figure(go.Pie(
        labels=df["outcome"], values=df["work_orders"], hole=0.62,
        marker=dict(colors=colors, line=dict(color=PLANE, width=2)),
        textinfo="label+percent", textfont=dict(color=INK, size=12),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        sort=False,
    ))
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig

