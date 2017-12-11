import os
import flask
import dash
from flask import Flask
from werkzeug.contrib.fixers import ProxyFix

server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/', csrf_protect=False)
app.server.wsgi_app = ProxyFix(app.server.wsgi_app)
app.title = 'Playground'
app.config.supress_callback_exceptions = True
app.scripts.config.serve_locally = True


# Add custom local css
static_files_directory = os.path.join(os.getcwd(), 'static')
static_files = [f for f in os.listdir(static_files_directory) if os.path.isfile(os.path.join(static_files_directory, f))]
# print(static_files)
static_route = '/static/'


@app.server.route('{}<file>'.format(static_route))
def serve_static_files(file):
    if stylesheet not in static_files:
        raise Exception('"{}" is excluded from the allowed static files'.format(file))
    return flask.send_from_directory(static_files_directory, file)


for stylesheet in ['dash_style.css', 'loading_style.css']:
    app.css.append_css({"external_url": "/static/{}".format(stylesheet)})


# from flask_caching import Cache
# CACHE_CONFIG = {
#     'CACHE_TYPE': 'simple',
# }
# cache = Cache()
# cache.init_app(app.server, config=CACHE_CONFIG)
