import cv2
import numpy as np
from socketio import Client
from datetime import datetime

# === LangChain + FAISS ===
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint

# =========================
# 🔐 Replace with NEW token
HF_TOKEN = "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"

# === LLM Setup ===
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",   # 🔥 MUST
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256
)

# === Create Vector DB (RAG) ===
def create_vector_db():
    docs = [
        Document(page_content="Call ambulance immediately for severe accidents"),
        Document(page_content="Nearest hospital is Apollo Hospital"),
        Document(page_content="Provide first aid to injured victims"),
        Document(page_content="Inform police for traffic control"),
    ]

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# === Retrieve Context ===
def retrieve_context(query, db):
    results = db.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

# === LLM Response ===
def generate_llm_response(description, db):
    context = retrieve_context(description, db)

    prompt = f"""
    Accident detected: {description}

    Context:
    {context}

    Provide:
    - Severity
    - Emergency actions
    """

    try:
        return llm(prompt)
    except:
        return """Severity: High
Call emergency services (108)
Provide first aid immediately"""

# === SocketIO Setup ===
socketio = Client()
socketio.connect('http://127.0.0.1:5000')

def send_alert(status, speeds, timestamp, location, llm_output):
    socketio.emit('update_status', {
        'status': status,
        'vehicle_speeds': speeds,
        'timestamp': timestamp,
        'location': location,
        'llm_output': llm_output
    })

# === Speed Calculation ===
def calculate_speed(prev_center, current_center, frame_interval):
    distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
    return distance / frame_interval

# === Simple Centroid Tracking (FIXED) ===
def match_vehicles(prev_centers, current_centers):
    new_tracked = {}
    for i, curr in enumerate(current_centers):
        min_dist = float('inf')
        matched_id = None

        for vid, prev in prev_centers.items():
            dist = np.linalg.norm(np.array(curr) - np.array(prev))
            if dist < min_dist and dist < 50:  # threshold
                min_dist = dist
                matched_id = vid

        if matched_id is not None:
            new_tracked[matched_id] = curr
        else:
            new_tracked[len(prev_centers) + i] = curr

    return new_tracked

# === Crash Detection ===
def detect_crash(vehicles, collision_frames):
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            x1, y1, x2, y2 = vehicles[i]
            x3, y3, x4, y4 = vehicles[j]

            if x1 < x4 and x3 < x2 and y1 < y4 and y3 < y2:
                key = f"{i}_{j}"
                collision_frames[key] = collision_frames.get(key, 0) + 1

                if collision_frames[key] > 5:
                    return True, {"Vehicle A": "Collision", "Vehicle B": "Collision"}
            else:
                collision_frames[f"{i}_{j}"] = 0

    return False, {}

# === MAIN ===
def main():
    cap = cv2.VideoCapture("cr.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    frame_interval = 1 / fps

    prev_frame = None
    tracked_vehicles = {}
    collision_frames = {}

    db = create_vector_db()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, 2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_centers = []
            vehicles = []
            speeds = {}

            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                current_centers.append(center)
                vehicles.append((x, y, x+w, y+h))

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            new_tracked = match_vehicles(tracked_vehicles, current_centers)

            for vid, center in new_tracked.items():
                if vid in tracked_vehicles:
                    speed = calculate_speed(tracked_vehicles[vid], center, frame_interval)
                    speeds[vid] = speed

                    cv2.putText(frame, f"ID:{vid} Speed:{speed:.2f}",
                                center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            tracked_vehicles = new_tracked

            accident, _ = detect_crash(vehicles, collision_frames)

            if accident:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                location = "Lat:12.9716, Lon:77.5946"

                description = "Vehicle collision detected"

                llm_output = generate_llm_response(description, db)

                print("\n🚨 LLM RESPONSE:\n", llm_output)

                send_alert("Accident Detected", speeds, timestamp, location, llm_output)

        cv2.imshow("AI System", frame)
        prev_frame = gray

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === RUN ===
if __name__ == "__main__":
    main()