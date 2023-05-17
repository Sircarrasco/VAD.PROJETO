from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Dash(__name__)

app.layout = html.Div(
    className="population_graph",
    children=[
        html.Div(
            id='population',
            children=[
                dcc.Graph(id='population_plot'),
                dcc.Dropdown(
                    id="dropdown",
                    options=[{'label': x, 'value': x}
                        for x in [
                            'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra',
                            'Angola', 'Anguilla', 'Antigua And Barbuda', 'Argentina',
                            'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',
                            'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
                            'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia',
                            'Bosnia And Herzegovina', 'Botswana', 'Brazil',
                            'British Virgin Islands', 'Brunei', 'Bulgaria', 'Burkina Faso',
                            'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Chad',
                            'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Cook Islands',
                            'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus',
                            'Czech Republic', 'Denmark', 'Djibouti', 'Dominica',
                            'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
                            'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia',
                            'Falkland Islands', 'Faroe Islands', 'Fiji', 'Finland', 'France',
                            'French Guiana', 'French Polynesia', 'Gabon', 'Gambia', 'Georgia',
                            'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland',
                            'Guadeloupe', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana',
                            'Haiti', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India',
                            'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Isle Of Man', 'Israel',
                            'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jersey', 'Jordan',
                            'Kazakhstan', 'Kenya', 'Kosovo (Disputed Territory)', 'Kuwait',
                            'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia',
                            'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar',
                            'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius',
                            'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro',
                            'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
                            'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand',
                            'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia',
                            'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea',
                            'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
                            'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russia', 'Rwanda',
                            'Saint Helena', 'Saint Kitts And Nevis', 'Saint Lucia',
                            'Saint Vincent And The Grenadines', 'Samoa', 'San Marino',
                            'Sao Tome And Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
                            'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten',
                            'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
                            'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka',
                            'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria',
                            'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste',
                            'Togo', 'Tonga', 'Trinidad And Tobago', 'Tunisia', 'Turkey',
                            'Turkmenistan', 'Turks And Caicos Islands', 'Tuvalu', 'Uganda',
                            'Ukraine', 'United Arab Emirates', 'United Kingdom',
                            'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
                            'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia',
                            'Zimbabwe'
                        ]
                    ],
                    value='Portugal',
                    clearable=False,
                    ),
            
                ]),
        
    ]
)

@app.callback(
        Output('population_plot', 'figure'),
        [Input('dropdown', 'value')]
)
def plot_population(country):
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
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
