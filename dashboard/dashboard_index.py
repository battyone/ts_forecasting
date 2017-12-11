import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from dashboard.app import app
from dashboard.pages.index import render_index_layout
import dashboard.pages.noise_analysis

app.layout = html.Div([
    # header,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    # html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),
    # html.Div(id='intermediate_variable', style={'display': 'none'}),
])


# Update the index page
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return render_index_layout()
    elif pathname == '/noise_analysis':
        return dashboard.pages.noise_analysis.render_noise_analysis_layout()
    else:
        return render_index_layout()


if __name__ == '__main__':
    app.run_server(port=8050)
