import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


# def generate_lr_plot(data, title):
#     x_title = data.columns[0]
#     y_title = data.columns[1]
    
#     # Linear regression
#     X, y = data[x_title].values.reshape(-1, 1), data[y_title].values
#     reg = LinearRegression().fit(X, y)
#     beta = reg.coef_[0]
#     const = reg.intercept_
#     corr = np.corrcoef(data[x_title], data[y_title])[0, 1]
#     resids = y - reg.predict(X)
    
#     x_vals = np.linspace(X.min(), X.max(), 100)
#     y_vals = reg.predict(x_vals.reshape(-1, 1))

#     # Plot
#     left_sub_title = title
#     right_sub_title = f"Residuals of {title}"

#     fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06, 
#                         subplot_titles=[left_sub_title, right_sub_title])

#     fig.add_annotation(xref='paper', yref='paper', x=0, y=-0.35, showarrow=False,
#                    text=f"Constant = {const:.3f}; Beta = {beta:.3f}; Corr = {corr:.2f}",
#                    font=dict(size=14))

#     # Figure 1
#     scatter_fig = px.scatter(x=data[x_title], y=data[y_title])
#     scatter_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', 
#                              hoverinfo='skip', line=dict(color='black')))
#     scatter_fig.update_layout(
#         showlegend=False,
#         margin=dict(l=0, r=0, t=0, b=0)
#     )

#     # Figure 2
#     resid_fig = go.Figure()
#     resid_fig.add_trace(go.Scatter(x=data.index, y=resids, mode='lines', name='Resids',
#                                    line=dict(color='black'), showlegend=False))
#     resid_fig.add_trace(go.Scatter(x=data.index, y=[0]*len(data), mode='lines',
#                                    name='Zero Line', line=dict(color='gray'), 
#                                    hoverinfo='skip', showlegend=False))

#     for trace in scatter_fig.data:
#         fig.add_trace(trace, row=1, col=1)

#     for trace in resid_fig.data:
#         fig.add_trace(trace, row=1, col=2)
    
#     fig.update_xaxes(title_text=x_title, row=1, col=1)
#     fig.update_yaxes(title_text=y_title, row=1, col=1)
#     fig.update_xaxes(title_text="Date", row=1, col=2)
    
#     # fig.update_layout(height=350, width=900, showlegend=False, 
#     #                   margin=dict(l=50, r=50, t=50, b=90))
#     fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=50, b=90))
    
#     fig.show()


def generate_lr_plot(data, title):
    x_title = data.columns[0]
    y_title = data.columns[1]

    # Linear regression
    X, y = data[x_title].values.reshape(-1, 1), data[y_title].values
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    const = reg.intercept_
    corr = np.corrcoef(data[x_title], data[y_title])[0, 1]
    resids = y - reg.predict(X)

    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = reg.predict(x_vals.reshape(-1, 1))

    # Subplot layout
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.06,
        subplot_titles=[title, f"Residuals of {title}"]
    )

    # Regression plot
    scatter_trace = go.Scatter(
        x=data[x_title], y=data[y_title], mode='markers', name='Data',
        marker=dict(color='blue')
    )
    line_trace = go.Scatter(
        x=x_vals, y=y_vals, mode='lines', name='Fit',
        line=dict(color='black'), hoverinfo='skip'
    )

    # Residuals plot
    resid_trace = go.Scatter(
        x=data.index, y=resids, mode='lines', name='Residuals',
        line=dict(color='black'), showlegend=False
    )
    zero_line = go.Scatter(
        x=data.index, y=[0]*len(data), mode='lines', name='Zero',
        line=dict(color='gray', dash='dash'), hoverinfo='skip', showlegend=False
    )

    # Add traces
    fig.add_trace(scatter_trace, row=1, col=1)
    fig.add_trace(line_trace, row=1, col=1)
    fig.add_trace(resid_trace, row=1, col=2)
    fig.add_trace(zero_line, row=1, col=2)

    # Axis labels
    fig.update_xaxes(title_text=x_title, row=1, col=1)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)

    # Annotation
    fig.add_annotation(
        xref='paper', yref='paper', x=0, y=-0.3, showarrow=False,
        text=f"Constant = {const:.3f}; Beta = {beta:.3f}; Corr = {corr:.2f}",
        font=dict(size=14)
    )

    # Layout
    fig.update_layout(
        autosize=True,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=90),
    )

    return fig