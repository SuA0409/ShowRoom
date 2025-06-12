from flask import Flask, jsonify, request, make_response, render_template
import os
import requests
import json
from pyngrok import ngrok
from flask_cors import CORS
import time
from io import BytesIO

json_path = '/content/drive/MyDrive/Final_Server/ngrok_path.json'
with open(json_path, 'r') as f:
    url = json.load(f)

# 각 서버 URL 할당
REVIEW_SERVER_URL = url.get("REVIEW_SERVER_URL") or ""
FAST3R_SERVER_URL = url.get("FAST3R_SERVER_URL") or ""
TWOD_SERVER_URL = url.get("TWOD_SERVER_URL") or ""

# 확인 출력
print(f"✅ REVIEW_SERVER_URL: {REVIEW_SERVER_URL}")
print(f"✅ FAST3R_SERVER_URL: {FAST3R_SERVER_URL}")
print(f"✅ TWOD_SERVER_URL: {TWOD_SERVER_URL}")

ngrok.set_auth_token("2whjTqF1XYhqkhqaiHpSEMlQ7w2_83j72xkR3qJcfxhzq5B8f")
print("💡 ngrok 연결 완료")

app = Flask(__name__, template_folder='templates')
CORS(app)

# 🔗 결과 JSON 저장 폴더
RESULTS_FOLDER = '/content/drive/MyDrive/Final_Server/main_server/results'
RECEIVED_FOLDER = '/content/drive/MyDrive/Final_Server/Input/Images'
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    start_time = time.time()  # ⭐️ 시작 시간 기록

    data = request.get_json()
    print(f"[⚡️] /analyze_review 요청 데이터: {data}")

    if not data or 'url' not in data:
        return jsonify({"status": "error", "message": "요청에 URL이 없습니다."}), 400

    url = data['url']
    print(f"[✓] 리뷰 분석 요청된 URL: {url}")

    try:
        headers = {'Content-Type': 'application/json'}
        print("[⚡️] 리뷰 분석 서버에 POST 요청 시작")
        review_response = requests.post(REVIEW_SERVER_URL + "/analyze_review",
                                        json={"url": url}, headers=headers, timeout=120)
        print(f"[⚡️] 리뷰 분석 서버 응답 코드: {review_response.status_code}")

        if review_response.status_code == 200:
            result = review_response.json()
            if result.get("status") == "success":
                room_id = result.get('room_id', 'unknown')

                # 📁 JSON 파일로 저장
                json_path = os.path.join(RESULTS_FOLDER, f"{room_id}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result['result'], f, ensure_ascii=False, indent=2)
                print(f"[✓] 결과 JSON 저장 완료: {json_path}")

                # 🔗 view_url 반환
                view_url = f"{public_url}/review/{room_id}"

                # ⭐️ 처리 시간 출력
                elapsed_time = time.time() - start_time
                print(f"[⏱️] 리뷰 분석 처리 시간: {elapsed_time:.2f}초")

                return jsonify({"status": "success", "view_url": view_url})
            else:
                error_msg = result.get("message", "Unknown error")
                print(f"[❌] 리뷰 분석 실패: {error_msg}")
                return jsonify({"status": "error", "message": "리뷰 분석 실패: " + error_msg}), 500
        else:
            print(f"[❌] 리뷰 분석 서버 응답 오류: {review_response.status_code}")
            return jsonify({"status": "error", "message": f"리뷰 분석 서버 오류: {review_response.status_code}"}), 500

    except Exception as e:
        print(f"[❌] 리뷰 분석 서버 요청 실패: {e}")
        return jsonify({"status": "error", "message": f"리뷰 분석 서버 통신 실패! {e}"}), 500


@app.route('/review/<room_id>')
def show_review(room_id):
    print(f"[⚡️] /review/{room_id} 호출됨")
    json_path = os.path.join(RESULTS_FOLDER, f"{room_id}.json")
    if not os.path.exists(json_path):
        print(f"[❌] 결과 JSON 파일이 없음: {json_path}")
        return "분석 결과가 없습니다.", 404

    with open(json_path, encoding='utf-8') as f:
        result = json.load(f)

    return render_template("review_result.html", room_id=room_id, topics=result['topics'])


# 3d_server resp,respone
@app.route('/3d_upload', methods=['POST'])
def upload_and_process():
    data = request.get_json()
    print(f"[⚡️] /3d_upload 요청 데이터: {data}")

    # 입력 검증
    if not data or 'images' not in data:
        return jsonify({"status": "error", "message": "요청에 이미지 정보가 없습니다."}), 400

    images = data['images']
    if len(images) < 1:
        return jsonify({"status": "error", "message": "저장할 이미지가 없습니다."}), 400

    print(f"[⚡️] 저장할 이미지 리스트: {images}")

    ## 이미지 byte로 변환
    files = []
    for i, url in enumerate(images):
        try:
            response = requests.get(url)
            response.raise_for_status()

            # 메모리 상의 파일 객체 생성
            img_file = BytesIO(response.content)
            img_file.name = f'{i}.jpg'

            # 원하는 형식으로 추가: (key, file-object)
            files.append((f'images{i}', img_file))

        except Exception as e:
            print(f"Failed to load image from {url}: {e}")

    # ✅ Fast3R 서버에 "폴더 전체 처리" 요청
    try:
        print("[⚡️] Fast3R에 요청 전송!")
        fast3r_response = requests.post(FAST3R_SERVER_URL + "/3d_upload", files=files, timeout=600)
        print(f"[⚡️] Fast3R 응답코드: {fast3r_response.status_code}")

        if fast3r_response.status_code == 200:
            fast3r_result = fast3r_response.json()
        else:
            return jsonify({"status": "error", "message": f"Fast3R 오류: {fast3r_response.status_code}"}), 500
    except Exception as e:
        print(f"[❌] Fast3R 요청 실패: {e}")
        return jsonify({"status": "error", "message": f"Fast3R 요청 실패: {e}"}), 500

    print("viser 시작")

    # ✅ Viser에 "시각화 요청" 전송
    try:
        print("[⚡️] Viser에 요청 전송!")
        viser_response = requests.post(FAST3R_SERVER_URL + "/viser", timeout=600)
        print(f"[⚡️] Viser 응답코드: {viser_response.status_code}")

        if viser_response.status_code == 200:
            viser_result = viser_response.json()
        else:
            return jsonify({"status": "error", "message": f"Viser 오류: {viser_response.status_code}"}), 500
    except Exception as e:
        print(f"[❌] Viser 요청 실패: {e}")
        return jsonify({"status": "error", "message": f"Viser 요청 실패: {e}"}), 500

    # ✅ 최종 응답 통합
    response_data = {
        "status": "success",
        "message": "이미지 저장, Fast3R 처리 및 Viser 요청까지 완료!",
        "fast3r_response": fast3r_result,
        "viser_response": viser_result
    }

    response = make_response(jsonify(response_data))
    response.headers['Content-Type'] = 'application/json'
    return response

# 2d_server resp,respone
@app.route('/2d_upload', methods=['POST'])
def request_2d_server():
    try:
        response_2d = requests.post(FAST3R_SERVER_URL + "/2d_upload", files=files, json=fast3r_response, timeout=600)
        print("🔔 2D 서버로 요청 시작!")

        if response_2d.status_code == 200:
            result_2d = response_2d.json()
            print("✅ 2D 서버 처리 완료:", result_2d)

            # ⭐️ 이어서 FAST3R 서버에 요청
            print("🔔 FAST3R 서버로 요청 시작!")
            response_3d = requests.post(FAST3R_SERVER_URL + "/3d_upload", timeout=600)

            if response_3d.status_code == 200:
                result_3d = response_3d.json()
                print("✅ FAST3R 처리 완료:", result_3d)

                # ⭐️ ⭐️ 이어서 VISER 요청 추가!
                print("🔔 VISER에 요청 시작!")
                response_viser = requests.post(FAST3R_SERVER_URL + "/viser", timeout=600)

                if response_viser.status_code == 200:
                    result_viser = response_viser.json()
                    print("✅ VISER 처리 완료:", result_viser)

                    return jsonify({
                        "status": "success",
                        "message": "2D, 3D, Viser까지 모두 완료!",
                        "2d_result": result_2d,
                        "3d_result": result_3d,
                        "viser_result": result_viser
                    })
                else:
                    return jsonify({"status": "error", "message": f"Viser 오류: {response_viser.text}"}), 500

            else:
                return jsonify({"status": "error", "message": "3D 서버 오류: " + response_3d.text}), 500
        else:
            return jsonify({"status": "error", "message": "2D 서버 오류: " + response_2d.text}), 500

    except Exception as e:
        print("❌ 2D 서버 요청 실패:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ✅ Flask 실행
if __name__ == '__main__':
    tunnel = ngrok.connect(5000)
    public_url = tunnel.public_url
    print(f"💡 Main 서버 ngrok 외부 URL: {public_url}")

    app.run(host='0.0.0.0', port=5000)