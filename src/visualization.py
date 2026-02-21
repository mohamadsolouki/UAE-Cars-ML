"""
Reusable visualization functions using Plotly.
All charts follow a consistent style with no emojis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# Consistent color palette
COLORS = px.colors.qualitative.Set2
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_ACCENT = "#2ca02c"
COLOR_WARN = "#d62728"

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=60),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def apply_layout(fig, title=None, height=500):
    """Apply consistent layout to a figure."""
    fig.update_layout(**LAYOUT_DEFAULTS, height=height)
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    return fig


def histogram(df, col, title=None, nbins=50, color=COLOR_PRIMARY, height=400):
    """Create a styled histogram."""
    fig = px.histogram(df, x=col, nbins=nbins, color_discrete_sequence=[color])
    fig.update_layout(bargap=0.05, yaxis_title="Count", xaxis_title=col)
    return apply_layout(fig, title=title or f"Distribution of {col}", height=height)


def box_plot(df, x, y, title=None, color=None, height=500):
    """Create a styled box plot."""
    fig = px.box(df, x=x, y=y, color=color, color_discrete_sequence=COLORS)
    return apply_layout(fig, title=title or f"{y} by {x}", height=height)


def violin_plot(df, x, y, title=None, height=500):
    """Create a styled violin plot."""
    fig = px.violin(df, x=x, y=y, box=True, color_discrete_sequence=COLORS)
    return apply_layout(fig, title=title or f"{y} by {x}", height=height)


def bar_chart(data, x, y, title=None, color=COLOR_PRIMARY, horizontal=False, height=400):
    """Create a styled bar chart from a Series or DataFrame."""
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation="h", color_discrete_sequence=[color])
    else:
        fig = px.bar(data, x=x, y=y, color_discrete_sequence=[color])
    return apply_layout(fig, title=title, height=height)


def scatter_plot(df, x, y, color=None, title=None, trendline=None, height=500, hover_data=None):
    """Create a styled scatter plot."""
    fig = px.scatter(
        df, x=x, y=y, color=color,
        trendline=trendline,
        color_discrete_sequence=COLORS,
        hover_data=hover_data,
        opacity=0.6,
    )
    return apply_layout(fig, title=title or f"{y} vs {x}", height=height)


def correlation_heatmap(df, columns, title="Correlation Matrix", height=600):
    """Create a correlation heatmap for numerical columns."""
    corr = df[columns].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    return apply_layout(fig, title=title, height=height)


def pie_chart(values, names, title=None, height=400):
    """Create a styled pie chart."""
    fig = px.pie(values=values, names=names, color_discrete_sequence=COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_layout(fig, title=title, height=height)


def sunburst_chart(df, path, values=None, title=None, height=500):
    """Create a styled sunburst chart."""
    fig = px.sunburst(df, path=path, values=values, color_discrete_sequence=COLORS)
    return apply_layout(fig, title=title, height=height)


def line_chart(df, x, y, color=None, title=None, height=400):
    """Create a styled line chart."""
    fig = px.line(df, x=x, y=y, color=color, color_discrete_sequence=COLORS, markers=True)
    return apply_layout(fig, title=title or f"{y} over {x}", height=height)


def grouped_bar(df, x, y, color, title=None, barmode="group", height=500):
    """Create a grouped bar chart."""
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, color_discrete_sequence=COLORS)
    return apply_layout(fig, title=title, height=height)


def residual_plot(y_true, y_pred, title="Residual Plot", height=450):
    """Create a residuals vs predicted values plot."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color=COLOR_PRIMARY, opacity=0.5, size=4),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_WARN)
    fig.update_layout(xaxis_title="Predicted Values", yaxis_title="Residuals")
    return apply_layout(fig, title=title, height=height)


def predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual", height=450):
    """Create a predicted vs actual scatter plot with diagonal line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=COLOR_PRIMARY, opacity=0.5, size=4),
        name="Predictions",
    ))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color=COLOR_WARN, dash="dash"),
        name="Perfect Prediction",
    ))
    fig.update_layout(xaxis_title="Actual Values", yaxis_title="Predicted Values")
    return apply_layout(fig, title=title, height=height)


def feature_importance_chart(names, importances, title="Feature Importance", height=500, top_n=20):
    """Create a horizontal bar chart of feature importances."""
    df = pd.DataFrame({"Feature": names, "Importance": importances})
    df = df.nlargest(top_n, "Importance").sort_values("Importance")
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    return apply_layout(fig, title=title, height=height)


def confusion_matrix_heatmap(cm, labels, title="Confusion Matrix", height=500):
    """Create a confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(tickangle=45),
    )
    return apply_layout(fig, title=title, height=height)


def elbow_plot(k_range, inertias, title="Elbow Method for Optimal K", height=400):
    """Create an elbow plot for K-means clustering."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=inertias, mode="lines+markers",
        marker=dict(color=COLOR_PRIMARY, size=8),
        line=dict(color=COLOR_PRIMARY),
    ))
    fig.update_layout(xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
    return apply_layout(fig, title=title, height=height)


def silhouette_plot(k_range, scores, title="Silhouette Scores by K", height=400):
    """Create a silhouette score plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=scores, mode="lines+markers",
        marker=dict(color=COLOR_ACCENT, size=8),
        line=dict(color=COLOR_ACCENT),
    ))
    fig.update_layout(xaxis_title="Number of Clusters (K)", yaxis_title="Silhouette Score")
    return apply_layout(fig, title=title, height=height)


def cluster_scatter_2d(x, y, labels, title="Cluster Visualization (PCA)", height=500):
    """Create a 2D scatter plot of clusters."""
    df = pd.DataFrame({"PC1": x, "PC2": y, "Cluster": labels.astype(str)})
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Cluster",
        color_discrete_sequence=COLORS, opacity=0.7,
    )
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return apply_layout(fig, title=title, height=height)
