from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import os
import pandas as pd
import numpy as np

app = Dash(__name__)

app.layout = html.Div(
    className='filters',
    children = [
        html.Div(
            className='filter_item',
            children=[
                html.P('Security Index',className='filter_sec'),
                dcc.RangeSlider(
                    min=0, max=100,
                    value = [0, 100],
                    id = 'security_index'
                ),
            ]
        ),
        html.Div(
            className='filter_item',
            children=[
                html.P('Quality Index', className='filter_sec'),
                dcc.RangeSlider(
                    min=0, max=100,
                    value = [0, 100],
                    id = 'quality_index'
                ),
            ]
        ),
        html.Div(
            className ='filter_item',
            children=[
                html.P('No. of UNESCO properties',className='filter_sec'),
                dcc.RangeSlider(
                    min=0, max=100,
                    value = [0, 58],
                    id = 'unesco_props'
                ),
            ]
        ),
        html.Div(
            className='filter_item',
            children = [
                html.P('Cost category', className='filter_sec'),
                dcc.Dropdown(
                    id='type_cost',
                    options = [{'label' : l, 'value': v} 
                               for l, v in zip(['High', 'Medium', 'Low'], 
                                               ['average_cost_rich', 'average_cost_medium', 'average_cost_lower']
                                               )
                    ],
                    value = 'average_cost_rich',
                    clearable=False
                ),
            ]
        ),
        dcc.Graph(id='graph'),
        dcc.Store(id='intermediate-value')
    ]
)

@app.callback(Output('intermediate-value', 'data'), 
              [Input('unesco_props', 'value'), Input('quality_index', 'value'), 
               Input('security_index', 'value'), Input('type_cost', 'value')])
def filter_data(unesco_props, quality_index, security_index, type_cost):
    path = os.path.join('data', 'dataset_alpha.csv')
    data = pd.read_csv(path)
    # define costs
    data['average_cost_rich'] = 2 * data.x2 + data.x24 + data.x48 + data.x30 + 60 * data.x37 + data.x38 + data.x6 + data.x23
    data['average_cost_medium'] = 2 * data.x3 + data.x4 + data.x49 + data.x28 + 30 * data.x37 + data.x38 + data.x6 + data.x23
    data['average_cost_lower'] = 2 * data.x1 + data.x49 + data.x8 + 10 * data.x37 + data.x23

    data['cost'] = data[type_cost]
    print(type_cost)
    
    # filter data according to the users preferences
    filtered_data = data.loc[
        (data.safety_index >= security_index[0]) & (data.safety_index <= security_index[1]) &
        (data.quality_of_life >= quality_index[0]) & (data.quality_of_life <= quality_index[1]) &
        (data.unesco_props >= unesco_props[0]) & (data.unesco_props <= unesco_props[1])
        # TODO: add cost
    ]
    return filtered_data.to_json(date_format='iso', orient='split')

@app.callback(
    Output('graph', 'figure'), [Input('intermediate-value', 'data')]
)
def plot_data(json_data):
    data = pd.read_json(json_data, orient='split')
    fig = go.Figure(data=[
        go.Scatter(
            x = data['unesco_props'],
            y = data['unesco_props']
        ),
        go.Scatter(
            x = data['quality_of_life'],
            y = data['quality_of_life']
        ),
        go.Scatter(
            x = data['safety_index'],
            y = data['safety_index']
        )
    ]
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
