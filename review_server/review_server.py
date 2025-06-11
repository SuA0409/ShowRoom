# review_server code
import os
import sys
BASE_DIR = '/content/drive/MyDrive/review_server'
sys.path.append(BASE_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import os, re, json
from review_analysis import run_topic_model_on_room

# ngrok 인증 토큰 설정
ngrok.set_auth_token("2xwkthyPz15CsSbartjgnt9aQde_3RoEvuB7Mz7oHHzuDJFia")

# Flask 앱 생성
app = Flask(__name__)
CORS(app)

# 리뷰 분석 결과 JSON 저장 폴더
RESULT_FOLDER = './results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ngrok으로 외부 접속 URL 생성
public_url = ngrok.connect(5000).public_url
print(f"✅ 리뷰 분석 서버 ngrok URL: {public_url}")

# 리뷰 분석 요청을 처리하는 엔드포인트
@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    # 요청된 JSON 데이터에서 숙소 URL 추출
    data = request.get_json()
    url = data.get('url')
    print("🔍 리뷰 분석 요청받은 URL:", url)

    # URL이 없으면 에러 반환
    if not url:
        return jsonify({"status": "error", "message": "숙소 URL이 없습니다."}), 400

    try:
        # URL에서 숙소 ID 추출 (정규표현식 사용)
        room_id_match = re.search(r'/rooms/(\d+)', url)
        if not room_id_match:
            return jsonify({"status": "error", "message": "유효한 숙소 URL이 아닙니다."}), 400
        room_id = room_id_match.group(1)

        # 리뷰 분석 함수 실행
        result = run_topic_model_on_room(url)

        # 분석 결과를 JSON 파일로 저장
        save_path = os.path.join(RESULT_FOLDER, f"{room_id}.json")
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        # 분석 결과를 JSON 형태로 반환
        return jsonify({
            "status": "success",
            "room_id": room_id,
            "result": result
        })

    except Exception as e:
        # 분석 실패 시 에러 메시지 반환
        print(f"[!] 리뷰 분석 실패: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ✅ Flask 서버 시작
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
