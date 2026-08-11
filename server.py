from flask import Flask, request, jsonify, send_from_directory, abort
import threading
import os
import uuid

import ai_core

app = Flask(__name__, static_folder='.')

# In-memory session chat histories
session_histories = {}


def get_session_history(session_id: str):
    if session_id not in session_histories:
        session_histories[session_id] = [{"role": "system", "content": "You are a helpful, friendly, and concise voice assistant about music. Give brief, direct answers suited for spoken responses."}]
    return session_histories[session_id]


@app.route('/')
def index():
    # Serve the main index page (index.html)
    return send_from_directory('.', 'index.html')


@app.route('/chat')
def chat_page():
    # Serve the chat UI
    return send_from_directory('.', 'chat.html')


# Generic helpers to serve other index pages explicitly if requested
@app.route('/index_2.html')
def index2():
    return send_from_directory('.', 'index_2.html')


@app.route('/index_3.html')
def index3():
    return send_from_directory('.', 'index_3.html')


@app.route('/index_4.html')
def index4():
    return send_from_directory('.', 'index_4.html')


@app.route('/index_5.html')
def index5():
    return send_from_directory('.', 'index_5.html')


@app.route('/api/send_message', methods=['POST'])
def send_message():
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    if not message:
        return jsonify({'error': 'empty message'}), 400

    chat_history = get_session_history(session_id)
    # Generate reply (this will mutate chat_history)
    reply = ai_core.generate_text_reply(message, chat_history)
    return jsonify({'reply': reply, 'session_id': session_id})


@app.route('/api/run_main', methods=['POST'])
def run_main():
    data = request.get_json() or {}
    duration = int(data.get('duration', 5))
    session_id = data.get('session_id', 'default')

    chat_history = get_session_history(session_id)

    # Run the voice pipeline synchronously (records on server for 'duration')
    try:
        result = ai_core.run_voice_once(duration=duration, chat_history=chat_history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Update stored chat history (ai_core.generate_llm_response already mutated it)
    session_histories[session_id] = chat_history

    # Return transcription, ai response and tts filename for client to play
    return jsonify({'transcription': result['transcription'], 'ai_response': result['ai_response'], 'tts_file': result['tts_file'], 'session_id': session_id})


@app.route('/tts/<path:filename>')
def tts_file(filename):
    # Serve the generated TTS file if it exists
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return abort(404)


# Generic static file serving for other assets/pages (if file exists)
@app.route('/<path:filename>')
def serve_file(filename):
    # Prevent catching API routes
    if filename.startswith('api/') or filename.startswith('tts/'):
        return abort(404)
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return abort(404)


if __name__ == '__main__':
    # Optionally preload models in a background thread to reduce first-request latency
    threading.Thread(target=ai_core.init_models, daemon=True).start()
    app.run(host='0.0.0.0', port=7860, debug=True)
