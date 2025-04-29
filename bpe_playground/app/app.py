from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.routes import init_routes
    init_routes(app)  # Register the routes

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
