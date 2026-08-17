"""
server.py
=========
Puente de comunicacion entre el frontend (paginas web en /web_page) y el
backend de Python (main.py, ai_core.py, audio_procesing, etc).

Variables de estado compartido
-------------------------------
    button_is_pressed_ai    -> bool : estado del boton grabar/parar del chat de IA
    button_is_pressed_dash  -> bool : estado del boton grabar/parar del dashboard
    message_in              -> str  : ultimo texto recibido desde el frontend
    message_out             -> str  : ultimo texto que el backend quiere mostrar en el frontend

Como importarlas en otros modulos
----------------------------------
`from server import button_is_pressed_ai` copia el VALOR en ese momento.
Si luego ese modulo reasigna la variable localmente, el cambio NO se ve
reflejado aqui ni en el resto de la app (limitacion normal de Python con
tipos inmutables como bool/str). Por eso, para LEER o ESCRIBIR el valor
real y compartido entre hilos (Flask corre las peticiones en su propio
hilo), usa siempre el modulo completo y sus funciones:

    import server

    server.set_button_ai(True)        # marcar "grabando"
    server.get_button_ai()            # leer estado actual
    server.set_message_out("Hola")    # texto que se mostrara en la interfaz
    server.get_message_in()           # ultimo texto recibido del frontend

Como conectarlo con una app Flask existente (ai_core.py)
----------------------------------------------------------
Este archivo define un Blueprint (`bp`) en vez de su propio servidor, para
que puedas registrarlo dentro del Flask app que ya sirve las paginas HTML,
sin abrir un segundo puerto:

    # dentro de ai_core.py, junto al resto de rutas
    from server import bp as state_bp
    app.register_blueprint(state_bp)

Si prefieres correrlo como servidor independiente (por ejemplo mientras
desarrollas el frontend por separado), este archivo tambien se puede
ejecutar directamente: `python server.py` (ver bloque __main__ al final).
"""

import threading
from flask import Blueprint, Flask, jsonify, request

# --------------------------------------------------------------------------
# Estado compartido
# --------------------------------------------------------------------------
button_is_pressed_ai = False      # Boton grabar/parar del chat de IA
button_is_pressed_dash = False    # Boton grabar/parar del dashboard
message_in = ""                   # Texto recibido: frontend -> backend
message_out = ""                  # Texto a mostrar: backend -> frontend

_lock = threading.Lock()          # Protege el estado entre hilos (Flask + pipeline de voz)


# --------------------------------------------------------------------------
# Getters / Setters thread-safe -> usar estos desde otros modulos de Python
# --------------------------------------------------------------------------
def get_button_ai() -> bool:
    with _lock:
        return button_is_pressed_ai


def set_button_ai(value: bool) -> bool:
    """Fija el flag del boton de IA de forma explicita (True = grabando)."""
    global button_is_pressed_ai
    with _lock:
        button_is_pressed_ai = bool(value)
        return button_is_pressed_ai


def toggle_button_ai() -> bool:
    """Invierte el flag: 1ra pulsacion -> True (empieza a grabar),
    2da pulsacion -> False (detiene). Devuelve el nuevo estado."""
    global button_is_pressed_ai
    with _lock:
        button_is_pressed_ai = not button_is_pressed_ai
        return button_is_pressed_ai


def get_button_dash() -> bool:
    with _lock:
        return button_is_pressed_dash


def set_button_dash(value: bool) -> bool:
    global button_is_pressed_dash
    with _lock:
        button_is_pressed_dash = bool(value)
        return button_is_pressed_dash


def toggle_button_dash() -> bool:
    global button_is_pressed_dash
    with _lock:
        button_is_pressed_dash = not button_is_pressed_dash
        return button_is_pressed_dash


def get_message_in() -> str:
    with _lock:
        return message_in


def set_message_in(text: str) -> str:
    """Llamar desde el endpoint que recibe texto del frontend."""
    global message_in
    with _lock:
        message_in = text or ""
        return message_in


def get_message_out() -> str:
    with _lock:
        return message_out


def set_message_out(text: str) -> str:
    """Llamar desde main.py / ai_core.py cuando haya un texto nuevo
    (ej: transcripcion o respuesta del asistente) para mostrar en la web."""
    global message_out
    with _lock:
        message_out = text or ""
        return message_out


def get_state() -> dict:
    """Snapshot completo del estado, util para sincronizar la interfaz de una vez."""
    with _lock:
        return {
            "button_is_pressed_ai": button_is_pressed_ai,
            "button_is_pressed_dash": button_is_pressed_dash,
            "message_in": message_in,
            "message_out": message_out,
        }


# --------------------------------------------------------------------------
# Blueprint Flask: expone el estado anterior como API REST para el frontend
# --------------------------------------------------------------------------
bp = Blueprint("state", __name__)


@bp.route("/api/button_ai", methods=["POST"])
def api_button_ai_toggle():
    """El frontend llama aqui CADA VEZ que se presiona el boton grabar/parar
    del chat de IA (1ra vez -> True, 2da vez -> False)."""
    return jsonify({"button_is_pressed_ai": toggle_button_ai()})


@bp.route("/api/button_ai", methods=["GET"])
def api_button_ai_status():
    return jsonify({"button_is_pressed_ai": get_button_ai()})


@bp.route("/api/button_dash", methods=["POST"])
def api_button_dash_toggle():
    """Igual que /api/button_ai pero para el boton grabar/parar del dashboard."""
    return jsonify({"button_is_pressed_dash": toggle_button_dash()})


@bp.route("/api/button_dash", methods=["GET"])
def api_button_dash_status():
    return jsonify({"button_is_pressed_dash": get_button_dash()})


@bp.route("/api/message_in", methods=["POST"])
def api_message_in():
    """El frontend envia texto hacia el backend (ej: lo que escribio el usuario)."""
    data = request.get_json(silent=True) or {}
    text = data.get("message", "")
    return jsonify({"message_in": set_message_in(text)})


@bp.route("/api/message_out", methods=["GET"])
def api_message_out():
    """El frontend consulta (polling) el texto que el backend quiere mostrar."""
    return jsonify({"message_out": get_message_out()})


@bp.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(get_state())


# --------------------------------------------------------------------------
# Modo standalone: permite correr `python server.py` para probar la API
# sola, sin depender de ai_core.py. En produccion, preferir registrar `bp`
# dentro del Flask app existente (ver docstring arriba).
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(bp)
    app.run(host="0.0.0.0", port=5001, debug=True)
