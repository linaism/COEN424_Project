import os
from flask_cors import CORS
from flask import Flask


def create_app():
    # create and configure the app
    app = Flask(__name__)

    # enable CORS
    CORS(app, resources={r'/*': {'origins': '*'}})

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app


app = create_app()

if __name__ == "__main__":
    #    app = create_app()
    print(" Starting app...")
    app.run(host="0.0.0.0", port=5000)