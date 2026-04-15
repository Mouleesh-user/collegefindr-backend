"""WSGI shim for Render default start command compatibility."""


def _load_app():
    try:
        from app import app as flask_app
        return flask_app
    except ModuleNotFoundError:
        from backend.app import app as flask_app
        return flask_app


application = _load_app()
app = application
