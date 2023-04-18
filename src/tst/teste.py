from dash import Dash, dcc, html
from dash.dependencies import Input, Output 
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np 
import os
import dash_leaflet as dl
import plotly.express as px
import dash

path = os.path.join('..', 'data', 'teste.csv')
data = pd.read_csv(path)

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    dbc.Row(
        [
            dbc.Col([
                html.H1(children = "Travel destination selection dashboard",
                        style = {'textAlign': 'center'}),

                html.Div(
                        children=[
                            dbc.Button("World", color="primary", className="me-1 w-5", value="World", id="continent1"),
                            dbc.Button("Asia", color="secondary", className="me-1", value="Asia", id="continent2"),
                            dbc.Button("Europe", color="secondary", className="me-1", value="Europe", id="continent3"),
                            dbc.Button("North America", color="secondary", className="me-1", value="North America", id="continent4"),
                            dbc.Button("South America", color="secondary", className="me-1", value="South America", id="continent5")
                        ],
                        style={'textAlign': 'center'},
                        id="continent"
                    ),

                dcc.Graph(id="graph"),
                ], width=9),

            dbc.Col([
                html.Div("Right column",style={"background-color": "black"}),
                dcc.Graph(id="table"),
                ], width=3),
        ]
    )
    
    )

@app.callback(
    [Output("graph", "figure"),
     Output("table", "figure")],

    [Input('continent1', 'n_clicks'),
    Input('continent2', 'n_clicks'),
    Input('continent3', 'n_clicks'),
    Input('continent4', 'n_clicks'),
    Input('continent5', 'n_clicks')],

    [Input('continent1', 'value'),
    Input('continent2', 'value'),
    Input('continent3', 'value'),
    Input('continent4', 'value'),
    Input('continent5', 'value')])

# function to show world map with filters 
def map_filter(button1_clicks, button2_clicks,button3_clicks, button4_clicks, button5_clicks, button1_value, button2_value, button3_value, button4_value, button5_value):
    
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

    filter = "safety index"
    # ['africa', 'asia', 'europe', 'north america', 'south america', 'usa', 'world']
    data_filter = pd.DataFrame()

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

    if filter in ["average_cost_rich","average_cost_medium","average_cost_lower"]:
        data_filter = filter
    
    elif filter == "safety index":
        data_filter = "safety_index"

    elif filter == "unesco props":
        data_filter = "unesco_props"

    elif filter == "quality of life":
        data_filter = "quality_of_life" 

    elif filter == "total population":
        data_filter = "total_population"

    else:
        data_filter = "average_cost_rich"
    
    # Create a Plotly world map with country codes
    fig = px.choropleth(data_frame      = aux_data,
                        locations       = 'code',
                        locationmode    = 'ISO-3',
                        color_continuous_scale=['white', 'orange'], # Set color
                        color           = data_filter,
                        scope           = aux_scope,
                        )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Disable legend
    fig.update_layout(
        showlegend=False  # hide legend
    )
    # Disable hover effects
    fig.data[0].hovertemplate = None

    #fig.show()
    lista = aux_data[["country",data_filter]].nlargest(5,data_filter).reset_index(drop=True)
    lista.index = lista.index + 1

    fig2 = go.Figure(data=[go.Table(
    header=dict(values=['Rank', 'Country',"Cost"],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),

    cells=dict(values=[lista.index.tolist(),
                       lista['country'].values.tolist(), # 1st column
                       lista[data_filter],
                       ], # 2nd column
                        line_color='darkslategray',
                        fill_color='lightcyan',
                        align='left'))
    ])

    fig2.update_layout(width=500, height=400)

    print("lista: ",lista.index.tolist())
    return fig, fig2

if __name__ == "__main__":
    app.run_server(debug=True)