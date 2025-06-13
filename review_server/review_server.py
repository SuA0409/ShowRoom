"""
review_server.py
- 숙소 URL을 받아 리뷰를 수집하고 BERTopic으로 분석
- 결과 JSON으로 반환 및 저장
"""

import os
import sys

BASE_DIR = '/content/drive/MyDrive/review_server'
sys.path.append(BASE_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import os, re, json
from review_analysis import run_topic_model_on_room  # 리뷰 분석 함수


# ==========================================
# ngrok 인증 및 연결
# ==========================================
ngrok.set_auth_token("2yGSKnM6Tviku0bqCV7bRN5y7gn_rLmTrz5SsPvRgd62yS5b")

# ==========================================
# Flask 앱 생성 및 CORS 허용
# ==========================================
app = Flask(__name__)
CORS(app)

# 리뷰 분석 결과 JSON 저장 경로
RESULT_FOLDER = './results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ngrok 공개 URL 출력
public_url = ngrok.connect(5000).public_url
print(f"리뷰 분석 서버 ngrok URL: {public_url}")


# ==========================================
# 리뷰 분석 요청 처리 엔드포인트
# ==========================================
@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    """
    POST 요청으로 숙소 URL을 받아 리뷰 분석을 수행
    - 리뷰를 분석하여 토픽별로 분류
    - 결과를 JSON으로 반환하고 로컬 파일로도 저장
    """
    data = request.get_json()
    url = data.get('url')
    print("리뷰 분석 요청받은 URL:", url)

    if not url:
        return jsonify({"status": "error", "message": "숙소 URL이 없습니다."}), 400

    try:
        # 숙소 ID 추출 (파일명으로 사용)
        room_id_match = re.search(r'/rooms/(\d+)', url)
        if not room_id_match:
            return jsonify({"status": "error", "message": "유효한 숙소 URL이 아닙니다."}), 400
        room_id = room_id_match.group(1)

        # 리뷰 분석 실행
        result = run_topic_model_on_room(url)

        # 분석 결과를 JSON 파일로 저장
        save_path = os.path.join(RESULT_FOLDER, f"{room_id}.json")
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        # 최종 결과 반환
        return jsonify({
            "status": "success",
            "room_id": room_id,
            "result": result
        })

    except Exception as e:
        print(f"[!] 리뷰 분석 실패: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ==========================================
# Flask 앱 실행
# ==========================================
if __name__ == '__main__':
    """
    Flask 앱 실행 (로컬 호스트 + ngrok)
    """
    app.run(host='0.0.0.0', port=5000)
