from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os
import pandas as pd
import numpy as np
import dash
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as pg

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True
app.config.prevent_initial_callbacks = 'initial_duplicate'

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
                        html.Div(id='output')
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
              [Input('unesco_props', 'value'), Input('quality_index', 'value'), Input('security_index', 'value'), Input('total_population', 'value'), Input('GDP', 'value')])
def filter_data(unesco_props, quality_index, security_index, total_population, gdp):
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
        (data.unesco_props >= unesco_props[0]) & (data.unesco_props <= unesco_props[1]) & 
        (data.total_population >= total_population[0]) & (data.total_population <= total_population[1]) &
        (data.GDP >= gdp[0]) & (data.GDP <= gdp[1])
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
            html.Div(id='filters-info'),
            dcc.Graph(id='cost_by_continent'),
            html.Div(
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
    display='block'
    content = html.Div(
        children=[
            html.Div(
                className='filter_item',
                children=[
                    html.P('Security Index', className='filter_sec'),
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
                    html.P('No. UNESCO properties', className='filter_sec'),
                    dcc.RangeSlider(
                        min=0, max=60,
                        value = [0, 60],
                        id = 'unesco_props'
                    ),
                ]
            ),
            html.Div(
            className ='filter_item',
            children=[
                html.P('GDP',className='filter_sec'),
                dcc.RangeSlider(
                    min=0.04, max=17420,
                    value = [0.04, 17420],
                    id = 'GDP'
                    ),
                ]
            ),
            html.Div(
                className ='filter_item',
                children=[
                    html.P('Population',className='filter_sec'),
                    dcc.RangeSlider(
                        min=1.120400e+04, max=1.412360e+09,
                        value = [1.120400e+04, 1.412360e+09],
                        id = 'total_population'
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

    
    d = dict(
    type='choropleth',
    locations = aux_data['code'],
    z=aux_data[map_variable],
    text=aux_data['country'],
    colorscale=["white",'orange']   
    )

    layout = dict(
                title = 'Meal, Inexpensive Restaurant (USD)',
                geo=dict(
                scope=aux_scope
            )
                
                )
    x = pg.Figure(data = [d], 
                layout = layout)

    return x

@app.callback(
    Output('cost_by_continent', 'figure'),
    [Input('intermediate-value', 'data'), Input('map_variable', 'value')]
)
def display_average_by_country(json_data, map_variable):
    data = pd.read_json(json_data, orient='split')

    continent_data = data.groupby('continent').mean(numeric_only=True).reset_index()

    fig = go.Figure(
        data = go.Bar(
            x = continent_data['continent'],
            y = continent_data[map_variable]
        )
        )

    fig.update_layout(
        title = dict(
                text=f'Average per continent',
                font=dict(size=30),
                x=0.5
            ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

@app.callback(
    Output('variables_scatter', 'figure'),
    [Input('y_variable', 'value'), Input('x_variable', 'value'), Input('intermediate-value', 'data')]
)
def display_scatter(y_var, x_var, json_data):
    df = pd.read_json(json_data, orient='split')
    labels = ['Security', 'Quality Index','Total Population', 'GDP', 'Unesco Properties']
    variables = ['safety_index', 'quality_of_life', 'total_population', 'GDP', 'unesco_props']
    labels_map = {}
    for l, v in zip(labels, variables):
        labels_map[v] = l
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
                
            ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
     )
    return fig

@app.callback(
    Output('side_menu', 'children'),
    [Input('map-graph', 'clickData'),
     Input('side_menu', 'style')]
)
def display_click_data(clickData,content):
    if clickData is not None :
        country = clickData['points'][0]['text']
        print("deu?")
        #!
        path = os.path.join('data', 'dataset_alpha.csv')
        data = pd.read_csv(path)
        country_data = data[data.country == country]
        pop_columns = [col for col in country_data.columns if col.startswith('pop_') and not col.endswith('%')]
        pop_data = country_data[pop_columns].T.reset_index()
        pop_data.columns = ['type_pop', 'percentage']
        pop_data['gender'] = ''
        pop_data['age-range'] = ''
        for index, desc in enumerate(pop_data['type_pop']):
            if 'female' in desc:
                pop_data.loc[index, 'gender'] = 'Female'
                pop_data.loc[index, 'percentage'] = pop_data.loc[index, 'percentage'] * country_data['female_population'].values[0] / 100

            else:
                pop_data.loc[index, 'gender'] = 'Male'
                pop_data.loc[index, 'percentage'] = pop_data.loc[index, 'percentage'] * country_data['male_population'].values[0] / 100


            if '0_14' in desc:
                pop_data.loc[index, 'age-range'] = '0-14'
            elif '15_64' in desc:
                pop_data.loc[index, 'age-range'] = '15-64'

            else:
                pop_data.loc[index, 'age-range'] = '65+'


        
        fig = go.Figure(
            data=[
            go.Bar(name='Female', x=pop_data.loc[pop_data.gender == 'Female', 'age-range'], y=pop_data.loc[pop_data.gender == 'Female', 'percentage']),
            go.Bar(name='Male', x=pop_data.loc[pop_data.gender == 'Male', 'age-range'], y=pop_data.loc[pop_data.gender == 'Male', 'percentage'])
            ]
        )
        fig.update_layout(
            title=dict(
                text = 'Distribution of the population',
                x = 0.5
            )
        )
        fig.update_xaxes(
            title_text = 'Age'
        )
        fig.update_yaxes(
            title_text = 'Percentage of the population',
        )
        data = data.loc[data.country == country]
        
        #country = 'Portugal'
        country_data = data[data.country == country]

        meals = [1, 2, 3]
        market = [23, 25, 24, 27]
        transports = [28, 30, 33]
        internet = [37, 38]
        habitation = [48, 49]

        meals = [f'x{m}' for m in meals]
        market = [f'x{m}' for m in market]
        transports = [f'x{m}' for m in transports]
        internet = [f'x{m}' for m in internet]
        habitation = [f'x{m}' for m in habitation]

        costs = {
        'Meals' : ((country_data['x1'] + country_data['x2']/2 + country_data['x3']) / 3).values[0] * 2,
        'Market' : country_data[market].mean(axis=0).values[0],
        'Transports' : country_data[transports].mean(axis=0).values[0] * 3,
        'Telecommunications' : country_data[internet].mean(axis=0).values[0],
        'Accomodation' : country_data[habitation].mean(axis=0).values[0] / 30.437
        }

        values = list(costs.values())
        names = list(costs.keys())

        fig2 = go.Figure(
            go.Pie(
                labels = names,
                values = values
            )
        )

        fig2.update_layout(
            title = dict(
                text = '% of each cost for the total value',
                x = 0.5
            )
        )

        children=[
            html.Div(country),
            dcc.Graph(id='population_plot', figure = fig),
            dcc.Graph(id='population_plot2', figure = fig2),
            dbc.Button("Back", className="me-1 w-5", id="back"),
                ]
    
        return html.Div(children=children)
    else:
        children=[
            html.Button('Reset', id='filters-selected', n_clicks=0),
            html.Div(id='filters-info'),
            dcc.Graph(id='cost_by_continent'),
            html.Div(
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
                ]
    return html.Div(children=children)
    
if __name__ == '__main__':
    app.run_server(debug=True)
