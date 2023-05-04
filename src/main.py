from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash_extensions.enrich import MultiplexerTransform
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
                        html.H1(children = "Travel destination selection dashboard", id="title",
                        style = {'textAlign': 'center'}),

                        html.Div(
                                children=[
                                    dbc.Button("World", color="secondary", className="me-1 button_class", value="World", id="continent1"),
                                    dbc.Button("Asia", color="secondary", className="me-1 button_class", value="Asia", id="continent2"),
                                    dbc.Button("Europe", color="secondary", className="me-1 button_class", value="Europe", id="continent3"),
                                    dbc.Button("North America", color="secondary", className="me-1 button_class", value="North America", id="continent4"),
                                    dbc.Button("South America", color="secondary", className="me-1 button_class", value="South America", id="continent5"),
                                    dbc.Button("Africa", color="secondary", className="me-1 button_class", value="Africa", id="continent6")
                                ],
                                style={'textAlign': 'center'},
                                id="continent"
                            ),
                        html.Div(
                            id='map_variable_container',
                            children=[dcc.Dropdown(
                                id='map_variable',
                                options = [
                                    {'label' : l, 'value' : v}
                                    for l, v in zip(
                                        ['Higher Cost', 'Medium Cost', 'Lowest Cost', 'Security Index', 'Quality Index', 'UNESCO Properties', 'GDP', 'Population'],
                                        ['average_cost_rich', 'average_cost_medium', 'average_cost_lower', 'safety_index', 'quality_of_life', 'unesco_props', 'GDP', 'total_population']
                                    )
                                ],
                                clearable = False,
                                value='average_cost_medium'
                            )]
                        ),

                        dcc.Graph(id="map-graph"),
                        dcc.RangeSlider(
                            min=62, 
                            max=2257, 
                            step=1, 
                            value=[62 ,2256.08], 
                            id='map_slider',
                            marks=None,
                            className='slider_class'
                            ),
                        html.Div(
                            id='no_data_container',
                            children=[
                                html.Div(id='no_data'),
                                html.P('No data available', id='no_data_text')
                            ]
                        )
                        ],
                    
                ),
                dbc.Col(
                    id = 'side_menu',
                    width=4,style={"height": "100vh","overflow-y": "scroll"},
                    children = [
                        html.Div(
                            id='side_menu_content'
                        ),
                        html.Div(id='call_side_menu', style={'display' : 'none'}),
                    ]
                )
            ]
        )),
        
        dcc.Store(id='intermediate-value'),
        dcc.Store(id='button_value'),
        dcc.Store(id='filters-data', data={
            'quality_index': [0, 100],
            'security_index': [0, 100],
            'unesco_props' : [0, 60],
            'GDP' : [0.04, 17420],
            'total_population' : [1.120400e+04, 1.412360e+09]
        })
    ],
    style={
        "margin-bottom": "-50px"
    }
)


@app.callback(Output('intermediate-value', 'data', allow_duplicate=True), 
              Output('filters-data', 'data', allow_duplicate=True),
              [Input('unesco_props', 'value'), Input('quality_index', 'value'), Input('security_index', 'value'), Input('total_population', 'value'), Input('GDP', 'value')])
def filter_data(unesco_props, quality_index, security_index, total_population, gdp):
    path = os.path.join('data', 'dataset_alpha.csv')
    data = pd.read_csv(path)
    # define costs
    
    # filter data according to the users preferences
    filtered_data = data.loc[
        (data.safety_index >= security_index[0]) & (data.safety_index <= security_index[1]) &
        (data.quality_of_life >= quality_index[0]) & (data.quality_of_life <= quality_index[1]) &
        (data.unesco_props >= unesco_props[0]) & (data.unesco_props <= unesco_props[1]) & 
        (data.total_population >= total_population[0]) & (data.total_population <= total_population[1]) &
        (data.GDP >= gdp[0]) & (data.GDP <= gdp[1])
    ]
    return filtered_data.to_json(date_format='iso', orient='split'), {
            'quality_index': quality_index,
            'security_index': security_index,
            'unesco_props' : unesco_props,
            'GDP' : gdp,
            'total_population' : total_population
        }



@app.callback(
    Output('filters-info', 'children'), Output('filters-data', 'data'), Output('filters-selected', 'n_clicks'),
    Input('filters-selected', 'n_clicks'), Input('filters-data', 'data')
)
def show_filters(n_clicks, filter_data):
    if n_clicks % 2 != 0:
        filter_data = {
            'quality_index': [0, 100],
            'security_index': [0, 100],
            'unesco_props' : [0, 60],
            'GDP' : [0.04, 17420],
            'total_population' : [1.120400e+04, 1.412360e+09]
        }

    display='block'
    content = html.Div(
        children=[
            html.Div(
                className='filter_item',
                children=[
                    html.P('Security Index', className='filter_sec'),
                    dcc.RangeSlider(
                        min=0, max=100,
                        value = filter_data['security_index'],
                        id = 'security_index',
                        className='slider_class'
                    ),
                ]
            ),
            html.Div(
                className='filter_item',
                children=[
                    html.P('Quality Index', className='filter_sec'),
                    dcc.RangeSlider(
                        min=0, max=100,
                        value = filter_data['quality_index'],
                        id = 'quality_index',
                        className='slider_class'
                    ),
                ]
            ),
            html.Div(
                className ='filter_item',
                children=[
                    html.P('No. UNESCO properties', className='filter_sec'),
                    dcc.RangeSlider(
                        min=0, max=60,
                        value = filter_data['unesco_props'],
                        id = 'unesco_props',
                        className='slider_class'
                    ),
                ]
            ),
            html.Div(
            className ='filter_item',
            children=[
                html.P('GDP',className='filter_sec'),
                dcc.RangeSlider(
                    min=0.04, max=17420,
                    value = filter_data['GDP'],
                    id = 'GDP',
                    className='slider_class'
                    ),
                ]
            ),
            html.Div(
                className ='filter_item',
                children=[
                    html.P('Population',className='filter_sec'),
                    dcc.RangeSlider(
                        min=1.120400e+04, max=1.412360e+09,
                        value = filter_data['total_population'],
                        id = 'total_population',
                        className='slider_class'
                    ),
                ]
            ),
        ],
        style = {'display' : display}
    )
    return content, filter_data, 0


@app.callback(
    Output("map-graph", "figure"),
    Output("button_value",'data'),
    [Input('continent1', 'n_clicks'),
    Input('continent2', 'n_clicks'),
    Input('continent3', 'n_clicks'),
    Input('continent4', 'n_clicks'),
    Input('continent5', 'n_clicks'),
    Input('continent6', 'n_clicks'),
    Input('continent1', 'value'),
    Input('continent2', 'value'),
    Input('continent3', 'value'),
    Input('continent4', 'value'),
    Input('continent5', 'value'),
    Input('continent6', 'value'),
    Input('intermediate-value', 'data'), 
    Input('map_variable', 'value'),
    Input("button_value",'data'),
    Input("map_slider",'value')]
    )

# function to show world map with filters 
def map_view(button1_clicks, button2_clicks,button3_clicks, button4_clicks, button5_clicks, button6_clicks, button1_value, button2_value, button3_value, button4_value, button5_value, button6_value,json_data, map_variable, button_variable,map_slider):
    data = pd.read_json(json_data, orient='split')
    if button1_clicks is None and button2_clicks is None and button3_clicks is None and button4_clicks is None and button5_clicks is None:
        country = "World"
    else:
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'continent1':
            country = button1_value
        elif button_id == 'continent2':
            country = button2_value
        elif button_id == 'continent3':
            country = button3_value
        elif button_id == 'continent4':
            country = button4_value
        elif button_id == 'continent5':
            country = button5_value
        elif button_id == 'continent6':
            country = button6_value
        else:
            country = button_variable["country"]

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

    elif country == "Africa":
        aux_data = data[data["continent"] == country]
        aux_scope = "africa"
    else:
        aux_data = data
        aux_scope = "world"

    min_max_values = {
        'average_cost_lower' : [45, 2118],
        'average_cost_medium' : [60, 2257],
        'average_cost_rich' : [118, 5467],
        'safety_index' : [16, 87],
        'quality_of_life' : [33, 88],
        'unesco_props' : [0, 58],
        'GDP' : [0, 17420],
        'total_population' : [11204, 1412360000.0]
    }


    aux_data = aux_data.loc[ (aux_data[map_variable] <= map_slider[1]) & (aux_data[map_variable] >= map_slider[0])]
    fig = px.choropleth(data_frame      = aux_data,
                        locations       = 'code',
                        locationmode    = 'ISO-3',
                        color_continuous_scale=['yellow', 'red'], # Set color
                        color           = map_variable,
                        scope           = aux_scope,
                        hover_name      = 'country',
                        range_color= min_max_values[map_variable],
                        )

    fig.update_layout(
        showlegend=True,
        margin={"r":0,"t":0,"l":0,"b":0},
        plot_bgcolor ='white',
        coloraxis_colorbar_title_text = '',
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            lenmode='fraction',
            len=0.93,
            ticks='outside',
            # tickvals=min_max_values[map_variable],
            # ticktext=['Low', 'High'],
            # dtick=2
        ),
        
    )
    # Disable hover effects
    fig.data[0].hovertemplate = None

    return fig, {"country": country}

@app.callback(
    Output('cost_by_continent', 'figure'),
    [Input('intermediate-value', 'data'), Input('map_variable', 'value')]
)
def display_average_by_country(json_data, map_variable):
    data = pd.read_json(json_data, orient='split')

    continent_data = data.groupby('continent').mean(numeric_only=True).reset_index()

    # Define colors for each bar
    colors = ['#30426A', '#30426A', '#30426A', '#30426A', '#30426A', '#30426A']

    fig = go.Figure(
        data = go.Bar(
            x = continent_data['continent'],
            y = continent_data[map_variable],
            marker=dict(
            color=colors
    )
        )
        )

    fig.update_layout(
        title = dict(
                text=f'Average per continent',
                x=0.5,
                y=0.85
            ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
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
    #continent_colors = LabelEncoder().fit_transform(df['continent'])
    continent_colors = {'Asia': '#FF5733', 'Europe': '#FFC300', 'Africa': '#DAF7A6', 'South America': '#C70039',"North America": "#30426A" ,'Oceania': '#900C3F'}
    marker_colors = [continent_colors[continent] if continent in continent_colors else 'black' for continent in df['continent']]

    fig = px.scatter(df, y=y_var, x=x_var, color='continent')
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
     )
    fig.update_xaxes(
        title_text= labels_map[x_var]
    )
    fig.update_yaxes(
        title_text= labels_map[y_var]
    )
    return fig

@app.callback(
    Output('side_menu', 'children' , allow_duplicate=True),
    [Input('map-graph', 'clickData'),
    Input('side_menu', 'style'),
    Input('map_variable','value')]
)
def display_click_data(clickData,content,map_variable):
    
    if clickData is not None :
        country = clickData['points'][0]['hovertext']
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
            go.Bar(name='Female', x=pop_data.loc[pop_data.gender == 'Female', 'age-range'], y=pop_data.loc[pop_data.gender == 'Female', 'percentage'], marker=dict(color='#30426A')),
            go.Bar(name='Male', x=pop_data.loc[pop_data.gender == 'Male', 'age-range'], y=pop_data.loc[pop_data.gender == 'Male', 'percentage'], marker=dict(color='#636363'))
            ]
        )
        fig.update_layout(
            title=dict(
                text = 'Distribution of the population',
                x = 0.5,
                y=0.85
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        fig.update_xaxes(
            title_text = 'Age'
        )
        fig.update_yaxes(
            title_text = 'Percentage of the population',
        )
        
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
                values = values,
                marker=dict(colors=['#274382', '#276782', '#30426A', '#31B9C9', '#32B0CB'])
            )
        )

        fig2.update_layout(
            title = dict(
                text = '% of each cost for the total value',
                x = 0.5,
                y=0.85
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        quality_of_index = get_info_graph(data, country, 'quality_of_life', 'Quality of life', [0, 100], '%')
        security_index = get_info_graph(data, country, 'safety_index', 'Security Index', [0, 100], '%')
        gdp = get_info_graph(data, country, 'GDP', 'GDP', [0, 20000], 'Billions US $')
        cost = get_info_graph(data, country, 'cost_average', 'Daily Cost', [data['cost_average'].min(), data['cost_average'].max()], 'US $')
    
        if map_variable in ["average_cost_medium","average_cost_rich","average_cost_lower"]:
            data["rank"] = data[map_variable].rank()
            rank = data.loc[data["country"] == country, "rank"].values[0]
        else:
            data["oposite_rank"] = data[map_variable].rank(ascending=False)
            rank = data.loc[data["country"] == country, "oposite_rank"].values[0]
        children=[
            dbc.Button("Back", className="me-1 w-5", id="back", style = {"display": "block"}, value = "no"),
            html.Div(id='country_header', children=[
                html.H1(country, id="country_name"),
                html.H3(f'#{int(rank)} / {len(data)}', id="rank"),
            ]),
            html.P(children='Daily Cost', className='country_info_item'),
            dcc.Graph(figure=cost),
            html.P(children='QUALITY OF LIFE', className='country_info_item'),
            dcc.Graph(figure=quality_of_index),
            html.P('SECURITY INDEX', className='country_info_item'),
            dcc.Graph(figure=security_index),
            html.P('GDP', className='country_info_item'),
            dcc.Graph(figure=gdp),
            dcc.Graph(id='population_plot', figure = fig),
            dcc.Graph(id='population_plot2', figure = fig2),
            ]
    
        return html.Div(children=children)
    else:
        children= get_side_bar()
    return html.Div(children=children)


def get_info_graph(data, country, variable, display_name, ticks_range, ticks_name):
    
    value = data.loc[data.country == country, variable].values[0]
    fig = go.Figure(
        data=[
            go.Histogram(x=data[variable], marker=go.histogram.Marker(color='#30426A')),
        ]
    )
    fig.add_vline(x=value, 
                  line_width=3, 
                  line_color='red', 
                  annotation = dict(
                    align = 'right',
                    text = value.round(1),
                    font_size = 18,
                    font_color = 'red',
                  ))
    fig.update_layout(
        width = 450,
        height = 130,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        range = ticks_range,
        title_text = ticks_name,
        title_font_size = 12
    )
    return fig

def get_side_bar():
    children=[
                dbc.Button('Reset', className="me-1 button_class", id='filters-selected', n_clicks=0),
                html.Div(id='filters-info'),
                dcc.Graph(id='cost_by_continent'),
                html.Div(
                    className="scatter_plot",
                    children=[
                        html.Div(
                            dcc.Graph(id='variables_scatter'),
                        ),
                        html.Div(
                            id='scatter_vars',
                            children = [
                                dcc.Dropdown(
                                    id = 'x_variable',
                                    options = [{'label' : l, 'value': v} 
                                                for l, v in zip(
                                                    ['Security', 'Quality Index','Total Population', 'GDP', 'Unesco Properties'], 
                                                    ['safety_index', 'quality_of_life', 'total_population', 'GDP', 'unesco_props'])],
                                    value = 'quality_of_life',
                                    clearable=False
                                ),
                                html.P('VS'),
                                dcc.Dropdown(
                                    id = 'y_variable',
                                    options = [{'label' : l, 'value': v} 
                                                for l, v in zip(
                                                    ['Security', 'Quality Index','Total Population', 'GDP', 'Unesco Properties'], 
                                                    ['safety_index', 'quality_of_life', 'total_population', 'GDP', 'unesco_props'])],
                                    value = 'safety_index',
                                    clearable=False
                                )
                            ]
                        ),
                                
                    ])
                ]
    return children


@app.callback(
    Output('side_menu', 'children', allow_duplicate=True),  
    Output('back','style'),
    Input('back', 'n_clicks'))
def back_callback(back):
    if back is not None:
        children = get_side_bar()
        return html.Div(children=children), {'display': 'none'} 

# callback to update slider
@app.callback(
    Output('map_slider', 'min'),
    Output('map_slider', 'max'),
    Output('map_slider', 'value'),
    Output('map_slider', 'step'),
    Input('map_variable','value'))
def back_callback(value):
    path = os.path.join('data', 'dataset_alpha.csv')
    data = pd.read_csv(path)
    
    min = data[value].min()
    max = data[value].max()

    return min, max, [min,max], 1
    

if __name__ == '__main__':
    app.run_server(debug=True)