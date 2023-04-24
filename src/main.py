from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os
import pandas as pd
import numpy as np
import dash
import plotly.express as px

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    className = 'main_page',
    children = [
        html.Div(
            dbc.Row(
            [
                dbc.Col(
                    id = 'main_menu',
                    width = 8,
                    children = [
                        html.H1(children = "Travel destination selection dashboard",
                        style = {'textAlign': 'center'}),

                        html.Div(
                                children=[
                                    dbc.Button("World", color="secondary", className="me-1 w-5", value="World", id="continent1"),
                                    dbc.Button("Asia", color="secondary", className="me-1", value="Asia", id="continent2"),
                                    dbc.Button("Europe", color="secondary", className="me-1", value="Europe", id="continent3"),
                                    dbc.Button("North America", color="secondary", className="me-1", value="North America", id="continent4"),
                                    dbc.Button("South America", color="secondary", className="me-1", value="South America", id="continent5")
                                ],
                                style={'textAlign': 'center'},
                                id="continent"
                            ),
                        html.Div(
                            children=[dcc.Dropdown(
                                id='map_variable',
                                options = [
                                    {'label' : l, 'value' : v}
                                    for l, v in zip(
                                        ['Higher Cost', 'Medium Cost', 'Lowest Cost', 'Security Index', 'Quality Index', 'UNESCO Properties', 'GDP', 'Population'],
                                        ['average_cost_rich', 'average_cost_medium', 'average_cost_lower', 'safety_index', 'quality_of_life', 'unesco_props', 'GDP', 'total_population']
                                    )
                                ],
                                value='average_cost_medium'
                            )]
                        ),

                        dcc.Graph(id="map-graph"),
                        ]
                    
                ),
                dbc.Col(
                    id = 'side_menu',
                    width=4,
                    children = [
                        html.Div(
                            id='side_menu_content'
                        ),
                        html.Div(id='call_side_menu', style={'display' : 'none'})
                    ]
                )
            ]
        )),
        dcc.Store(id='intermediate-value')
    ]
)


@app.callback(Output('intermediate-value', 'data'), 
              [Input('unesco_props', 'value'), Input('quality_index', 'value'), Input('security_index', 'value')])
def filter_data(unesco_props, quality_index, security_index):
    path = os.path.join('data', 'dataset_alpha.csv')
    data = pd.read_csv(path)
    # define costs
    data['average_cost_rich'] = 2 * data.x2 + data.x24 + data.x48 + data.x30 + 60 * data.x37 + data.x38 + data.x6 + data.x23
    data['average_cost_medium'] = 2 * data.x3 + data.x4 + data.x49 + data.x28 + 30 * data.x37 + data.x38 + data.x6 + data.x23
    data['average_cost_lower'] = 2 * data.x1 + data.x49 + data.x8 + 10 * data.x37 + data.x23

    # filter data according to the users preferences
    filtered_data = data.loc[
        (data.safety_index >= security_index[0]) & (data.safety_index <= security_index[1]) &
        (data.quality_of_life >= quality_index[0]) & (data.quality_of_life <= quality_index[1]) &
        (data.unesco_props >= unesco_props[0]) & (data.unesco_props <= unesco_props[1])
        # TODO: add cost
    ]
    return filtered_data.to_json(date_format='iso', orient='split')


@app.callback(
    Output('side_menu_content', 'children'),
    Input('call_side_menu', 'children')
)
def side_menu(country):
    # filters button clicked?
    children=[
            html.Button('Reset', id='filters-selected', n_clicks=0),
            html.Div(id='filters-info')
    ]
    # if country:
    #     # country selected
    #     graphs_info = html.Div(
    #         #Country info

    #     )
    # else:
    #     # worldwide view
    #     graphs_info = html.Div(
    #         # menu with no country selected
    #     )
    # children.append(graphs_info)
    return html.Div(children=children)


@app.callback(
    Output('filters-info', 'children'),
    Input('filters-selected', 'n_clicks')
)
def show_filters(n_clicks):
    if n_clicks % 2 == 1:
        display = 'block'
    else:
        display = 'none'
    display='block'
    content = html.Div(
        children=[
            html.Div(
                className='filter_item',
                children=[
                    html.P('Security Index'),
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
                    html.P('Quality Index'),
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
                    html.P('No. of UNESCO properties'),
                    dcc.RangeSlider(
                        min=0, max=60,
                        value = [0, 60],
                        id = 'unesco_props'
                    ),
                ]
            ),

        ],
        style = {'display' : display}
    )
    return content


@app.callback(
    Output("map-graph", "figure"),

    [Input('continent1', 'n_clicks'),
    Input('continent2', 'n_clicks'),
    Input('continent3', 'n_clicks'),
    Input('continent4', 'n_clicks'),
    Input('continent5', 'n_clicks')],

    [Input('continent1', 'value'),
    Input('continent2', 'value'),
    Input('continent3', 'value'),
    Input('continent4', 'value'),
    Input('continent5', 'value')],
    Input('intermediate-value', 'data'), 
    Input('map_variable', 'value')
    )

# function to show world map with filters 
def map_filter(button1_clicks, button2_clicks,button3_clicks, button4_clicks, button5_clicks, button1_value, button2_value, button3_value, button4_value, button5_value, json_data, map_variable):
    data = pd.read_json(json_data, orient='split')
    if button1_clicks is None and button2_clicks is None and button3_clicks is None and button4_clicks is None and button5_clicks is None:
        country = "World"
    else:
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'continent1':
            #print(button1_value)
            country = button1_value
        elif button_id == 'continent2':
            #print(button2_value)
            country = button2_value
        elif button_id == 'continent3':
            #print(button3_value)
            country = button3_value
        elif button_id == 'continent4':
            #print(button4_value)
            country = button4_value
        elif button_id == 'continent5':
            #print(button5_value)
            country = button5_value

    if country == "Asia":
        aux_data = data[data["continent"] == country]
        aux_scope = "asia"

    elif country == "Europe":
        aux_data = data[data["continent"] == country]
        aux_scope = "europe"

    elif country == "North America":
        aux_data = data[data["continent"] == country]
        aux_scope = "north america"

    elif country == "South America":
        aux_data = data[data["continent"] == country]
        aux_scope = "south america"
    else:
        aux_data = data
        aux_scope = "world"

    
    # Create a Plotly world map with country codes
    fig = px.choropleth(data_frame      = aux_data,
                        locations       = 'code',
                        locationmode    = 'ISO-3',
                        color_continuous_scale=['white', 'orange'], # Set color
                        color           = map_variable,
                        scope           = aux_scope,
                        )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Disable legend
    fig.update_layout(
        showlegend=True
    )
    # Disable hover effects
    fig.data[0].hovertemplate = None

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
