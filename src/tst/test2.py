import dash
from dash import dcc
from dash import html
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import os 
import pandas as pd

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # GDP
    dcc.Graph(id='kde-plot_gdp'),
    dcc.Slider(
        id='z-slider_gdp',
        step=1000,
        disabled=True,
    ),
    # Estimated cost
    dcc.Graph(id='kde-plot_ec'),
    dcc.Slider(
        id='z-slider_ec',
        step=100,
        disabled=True,
    ),
    # Security Index
    dcc.Graph(id='kde-plot_si'),
    dcc.Slider(
        id='z-slider_si',
        step=10,
        disabled=True,
    ),
    # Quality Index
    dcc.Graph(id='kde-plot_qi'),
    dcc.Slider(
        id='z-slider_qi',
        step=10,
        disabled=True,
    ),
])

# Define the callback function to update the KDE plot
# GDP
@app.callback(
    Output('kde-plot_gdp', 'figure'),
    Output('z-slider_gdp', 'min'),
    Output('z-slider_gdp', 'max'),
    Output('z-slider_gdp', 'value'),
    Input('z-slider_gdp', 'value')
)
def update_kde_plot(z_value):
    country = "Portugal"
    filter = 'GDP'
    path = os.path.join('..', 'data', 'teste.csv')
    data = pd.read_csv(path)
    max_val = data[filter].max()
    min_val = data[filter].min()
    country_value = data[filter].loc[data["country"] == country].iloc[0]

    #! dps mudar
    if data[filter].isna().count() > 0:
        data[filter].fillna(0, inplace=True)

    column_data = data[filter].tolist()
    hist_data = [column_data]

    group_labels = ['GDP'] 

    fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
    fig.add_shape(type='circle',
              xref='x', yref='paper',
              x0=country_value-0.2, y0=0, x1=country_value+0.2, y1=1,
              fillcolor='red', line_color='red', opacity=0.5)
    return fig, min_val, max_val, country_value

# Estimated cost
@app.callback(
    Output('kde-plot_ec', 'figure'),
    Output('z-slider_ec', 'min'),
    Output('z-slider_ec', 'max'),
    Output('z-slider_ec', 'value'),
    Input('z-slider_ec', 'value')
)
def update_kde_plot(z_value):
    country = "Portugal"
    filter = 'average_cost_medium'
    path = os.path.join('..', 'data', 'teste.csv')
    data = pd.read_csv(path)
    max_val = data[filter].max()
    min_val = data[filter].min()
    country_value = data[filter].loc[data["country"] == country].iloc[0]

    #! dps mudar
    if data[filter].isna().count() > 0:
        data[filter].fillna(0, inplace=True)

    column_data = data[filter].tolist()
    hist_data = [column_data]

    group_labels = ['average_cost_medium'] 

    fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
    fig.add_shape(type='circle',
              xref='x', yref='paper',
              x0=country_value-0.2, y0=0, x1=country_value+0.2, y1=1,
              fillcolor='red', line_color='red', opacity=0.5)
    
    return fig, min_val, max_val, country_value

# Security Index
@app.callback(
    Output('kde-plot_si', 'figure'),
    Output('z-slider_si', 'min'),
    Output('z-slider_si', 'max'),
    Output('z-slider_si', 'value'),
    Input('z-slider_si', 'value')
)
def update_kde_plot(z_value):
    country = "Portugal"
    filter = 'safety_index'
    path = os.path.join('..', 'data', 'teste.csv')
    data = pd.read_csv(path)
    max_val = data[filter].max()
    min_val = data[filter].min()
    country_value = data[filter].loc[data["country"] == country].iloc[0]

    #! dps mudar
    if data[filter].isna().count() > 0:
        data[filter].fillna(0, inplace=True)

    column_data = data[filter].tolist()
    hist_data = [column_data]

    group_labels = ['safety_index'] 

    fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
    fig.add_shape(type='circle',
              xref='x', yref='paper',
              x0=country_value-0.2, y0=0, x1=country_value+0.2, y1=1,
              fillcolor='red', line_color='red', opacity=0.5)

    return fig, min_val, max_val, country_value

# Quality Index
@app.callback(
    Output('kde-plot_qi', 'figure'),
    Output('z-slider_qi', 'min'),
    Output('z-slider_qi', 'max'),
    Output('z-slider_qi', 'value'),
    Input('z-slider_qi', 'value')
)
def update_kde_plot(z_value):
    country = "Portugal"
    filter = 'quality_of_life'
    path = os.path.join('..', 'data', 'teste.csv')
    data = pd.read_csv(path)
    max_val = data[filter].max()
    min_val = data[filter].min()
    country_value = data[filter].loc[data["country"] == country].iloc[0]

    #! dps mudar
    if data[filter].isna().count() > 0:
        data[filter].fillna(0, inplace=True)

    column_data = data[filter].tolist()
    hist_data = [column_data]

    group_labels = ['quality_of_life'] 

    fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
    fig.add_shape(type='circle',
              xref='x', yref='paper',
              x0=country_value-0.2, y0=0, x1=country_value+0.2, y1=1,
              fillcolor='red', line_color='red', opacity=0.5)

    return fig, min_val, max_val, country_value


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
