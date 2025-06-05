from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from pyngrok import ngrok
import os, requests, json, subprocess, re

# ✅ 리뷰 분석 함수
from review.review_analysis import run_topic_model_on_room

# ngrok 토큰
ngrok.set_auth_token("2whjTqF1XYhqkhqaiHpSEMlQ7w2_83j72xkR3qJcfxhzq5B8f")

# Flask 초기화
app = Flask(__name__, template_folder='./review/templates')
CORS(app)

# 경로
RECEIVED_FOLDER = '/content/drive/MyDrive/server_test/Input/Images'
RESULT_FOLDER = '/content/drive/MyDrive/server_test/review/results'
os.makedirs(RECEIVED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ngrok 연결
public_url = ngrok.connect(5000).public_url
print(f"✅ ngrok URL: {public_url}")

# 1️⃣ 이미지 저장 + ST-RoomNet & rotate_and_inpainting 실행
@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    if not data or 'images' not in data:
        return jsonify({"status": "error", "message": "요청에 이미지 정보가 없습니다."}), 400

    images = data.get('images', [])
    if len(images) < 1:
        return jsonify({"status": "error", "message": "저장할 이미지가 없습니다."}), 400

    print("저장할 이미지 리스트:", images)

    # 기존 이미지 삭제
    for filename in os.listdir(RECEIVED_FOLDER):
        file_path = os.path.join(RECEIVED_FOLDER, filename)
        try:
            os.remove(file_path)
            print(f"[✓] 기존 이미지 삭제됨: {file_path}")
        except Exception as e:
            print(f"[!] 기존 이미지 삭제 실패: {file_path} - {e}")

    # 새 이미지 저장
    for idx, url in enumerate(images):
        try:
            response = requests.get(url)
            filename = f"{idx}.jpg"
            save_path = os.path.join(RECEIVED_FOLDER, filename)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"[✓] 저장됨: {save_path}")
        except Exception as e:
            print(f"[!] 이미지 다운로드 실패 ({url}): {e}")

    # ✅ ST-RoomNet 실행
    try:
        print("🔔 ST-RoomNet 실행 시작!")
        subprocess.run(
            ["python", "/content/drive/MyDrive/server_test/ST-RoomNet/ST_RoomNet.py"],
            check=True
        )
        print("✅ ST-RoomNet 실행 완료!")

        # ✅ rotate_and_inpainting.py 실행
        print("🔔 rotate_and_inpainting.py 실행 시작!")
        subprocess.run(
            ["python", "/content/drive/MyDrive/server_test/rotate_and_inpaint.py"],
            check=True
        )
        print("✅ rotate_and_inpainting.py 실행 완료!")

    except subprocess.CalledProcessError as e:
        print("❌ 모델 실행 중 오류:", e)
        return jsonify({"status": "error", "message": "모델 실행 오류"}), 500

    response = make_response(jsonify({"status": "success", "message": "저장 및 모델 실행 완료!"}))
    response.headers['Content-Type'] = 'application/json'
    return response

# 2️⃣ 리뷰 분석
@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    data = request.get_json()
    url = data.get('url')
    print("url:", url)
    if not url:
        return jsonify({"error": "숙소 URL이 없습니다."}), 400

    try:
        room_id = re.search(r'/rooms/(\d+)', url).group(1)
        save_path = f"{RESULT_FOLDER}/{room_id}.json"

        # ✅ 리뷰 분석
        result = run_topic_model_on_room(url)

        # 결과 저장
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        return jsonify({
            "status": "success",
            "view_url": f"{public_url}/review/{room_id}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3️⃣ 리뷰 결과 페이지
@app.route('/review/<room_id>')
def show_review(room_id):
    path = f"{RESULT_FOLDER}/{room_id}.json"
    if not os.path.exists(path):
        return "분석 결과가 없습니다.", 404

    with open(path, encoding='utf-8') as f:
        result = json.load(f)

    return render_template("review_result.html", room_id=room_id, topics=result['topics'])

# Flask 실행
app.run(port=5000)
