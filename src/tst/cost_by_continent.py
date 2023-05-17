from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Dash(__name__)

app.layout = html.Div(
    className="scatter_plot",
    children = [
        dcc.Graph(id='cost_by_continent'),
        dcc.Dropdown(
        id="dropdown",
        options=[{'label': x, 'value': x}
            for x in ['Gold', 'MediumTurquoise', 'LightGreen']
        ],
        value='Gold',
        clearable=False,
        ),
    ]
)

@app.callback(
    Output('cost_by_continent', 'figure'),
    [Input('dropdown', 'value')]
)
def display_scatter(var):
    path = os.path.join('data', 'dataset_alpha.csv')
    data = pd.read_csv(path)
    variables = ['x1', 'x28', 'x8', 'x38', 'x48']
    data['cost'] = data[variables].sum(axis=1)

    continent_data = data.groupby('continent').mean(numeric_only=True).reset_index()

    fig = go.Figure(
        data = go.Bar(
            x = continent_data['continent'],
            y = continent_data['cost']
        )
        )

    fig.update_layout(
        title = dict(
                text=f'Average cost of life per continent',
                font=dict(size=30),
                x=0.5
            )
        )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
