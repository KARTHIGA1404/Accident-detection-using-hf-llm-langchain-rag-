from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# === Flask App Setup ===
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")  # allow frontend connections

# === Route ===
@app.route('/')
def index():
    return render_template('index.html')

# === Socket Event ===
@socketio.on('update_status')
def handle_status_update(data):
    status = data.get('status')
    vehicle_speeds = data.get('vehicle_speeds', {})
    timestamp = data.get('timestamp')
    location = data.get('location')
    llm_output = data.get('llm_output')

    # === Debug logs (for backend) ===
    print("\n===== ALERT RECEIVED =====")
    print(f"Status: {status}")
    print(f"Vehicle Speeds: {vehicle_speeds}")
    print(f"Timestamp: {timestamp}")
    print(f"Location: {location}")
    print(f"LLM Output:\n{llm_output}")
    print("==========================\n")

    # === Send to frontend ===
    emit('status_update', {
        'status': status,
        'vehicle_speeds': vehicle_speeds,
        'timestamp': timestamp,
        'location': location,
        'llm_output': llm_output
    }, broadcast=True)

# === Run Server ===
if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO Server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)