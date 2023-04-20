from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Dash(__name__)

app.layout = html.Div(
    className="scatter_plot",
    children=[
        html.Div(
            id='scatter_horizontal',
            children=[
            html.Div(
                id = 'scatter_vertical',
                children = [
                    dcc.Graph(id='variables_scatter'),
                    dcc.Dropdown(
                        id = 'x_variable',
                        options = [{'label' : l, 'value': v} 
                                    for l, v in zip(
                                        ['Security', 'Quality Index','Total Population', 'GDP', 'Unesco Properties'], 
                                        ['safety_index', 'quality_of_life', 'total_population', 'GDP', 'unesco_props'])],
                        value = 'quality_of_life',
                        clearable=False

                    ),
                ]
            ),
            dcc.Dropdown(
                id = 'y_variable',
                options = [{'label' : l, 'value': v} 
                            for l, v in zip(
                                ['Security', 'Quality Index','Total Population', 'GDP', 'Unesco Properties'], 
                                ['safety_index', 'quality_of_life', 'total_population', 'GDP', 'unesco_props'])],
                value = 'safety_index',
                clearable=False
            ),
    ]),
        
    ]
)

@app.callback(
    Output('variables_scatter', 'figure'),
    [Input('y_variable', 'value'), Input('x_variable', 'value'),]
)
def display_scatter(y_var, x_var):

    labels = ['Security', 'Quality Index','Total Population', 'GDP', 'Unesco Properties']
    variables = ['safety_index', 'quality_of_life', 'total_population', 'GDP', 'unesco_props']
    labels_map = {}
    for l, v in zip(labels, variables):
        labels_map[v] = l

    path = os.path.join('data', 'dataset_alpha.csv')
    df = pd.read_csv(path)
    continent_colors = LabelEncoder().fit_transform(df['continent'])
    fig = go.Figure(
        data= go.Scatter(
            y = df[y_var],
            x = df[x_var],
            text = df['country'],
            mode ='markers',
            # marker_color = continent_colors
            # marker_size=df['total_population'],
            # marker_max_size=20
        ),
    )
    fig.update_layout(
        title = dict(
                text=f'{labels_map[y_var]} - {labels_map[x_var]}',
                # font=dict(size=24, weight='bold'),
                x=0.5,
                
            )
     )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
