#백업
from flask import Flask, jsonify, request, make_response, render_template
import os
import requests
import json
from pyngrok import ngrok
from flask_cors import CORS

# 🔗 외부 서버 주소
REVIEW_SERVER_URL = "https://9372-34-19-25-13.ngrok-free.app"
FAST3R_SERVER_URL = "https://5602-34-90-251-219.ngrok-free.app"
TWOD_SERVER_URL = "https://ce00-34-122-75-151.ngrok-free.app"

ngrok.set_auth_token("2whjTqF1XYhqkhqaiHpSEMlQ7w2_83j72xkR3qJcfxhzq5B8f")
print("💡 ngrok 연결 완료")

app = Flask(__name__, template_folder='templates')
CORS(app)

# 🔗 결과 JSON 저장 폴더
RESULTS_FOLDER = '/content/drive/MyDrive/Final_Server/main_server/results'
RECEIVED_FOLDER = '/content/drive/MyDrive/Final_Server/Input/Images' 
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ✅ 디버깅: ngrok URL 로그
public_url = ngrok.connect(5000).public_url
print(f"[⚡️] ngrok public_url: {public_url}")

@app.route('/analyze_review', methods=['POST'])
def analyze_review():
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
            print(f"[⚡️] 리뷰 분석 서버 응답 내용 (일부): {result.get('status')}, {result.get('room_id')}")

            if result.get("status") == "success":
                room_id = result.get('room_id', 'unknown')
                print(f"[✓] room_id: {room_id}")

                # 📁 JSON 파일로 저장
                json_path = os.path.join(RESULTS_FOLDER, f"{room_id}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result['result'], f, ensure_ascii=False, indent=2)
                print(f"[✓] 결과 JSON 저장 완료: {json_path}")

                # 🔗 JSON 기반으로 HTML 보여줄 URL 반환
                view_url = f"{public_url}/review/{room_id}"
                print(f"[⚡️] view_url 반환: {view_url}")
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

#3d_server resp,respone
@app.route('/3d_upload', methods=['POST'])
def upload():
    data = request.get_json()
    print(f"[⚡️] /upload 요청 데이터: {data}")

    if not data or 'images' not in data:
        return jsonify({"status": "error", "message": "요청에 이미지 정보가 없습니다."}), 400

    images = data.get('images', [])
    if len(images) < 1:
        return jsonify({"status": "error", "message": "저장할 이미지가 없습니다."}), 400

    print(f"[⚡️] 저장할 이미지 리스트: {images}")

    # 기존 이미지 삭제
    for filename in os.listdir(RECEIVED_FOLDER):
        file_path = os.path.join(RECEIVED_FOLDER, filename)
        try:
            os.remove(file_path)
            print(f"[✓] 기존 이미지 삭제됨: {file_path}")
        except Exception as e:
            print(f"[❌] 기존 이미지 삭제 실패: {file_path} - {e}")

    # 새 이미지 저장
    saved_files = []
    for idx, url in enumerate(images):
        try:
            response = requests.get(url)
            filename = f"{idx}.jpg"
            save_path = os.path.join(RECEIVED_FOLDER, filename)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"[✓] 저장됨: {save_path}")
            saved_files.append(save_path)
        except Exception as e:
            print(f"[❌] 이미지 다운로드 실패 ({url}): {e}")

    # FAST3R 서버 전송
    fast3r_responses = []
    for file_path in saved_files:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            try:
                fast3r_response = requests.post(FAST3R_SERVER_URL+"/3d_upload", files=files, timeout=60)
                print(f"[⚡️] FAST3R 응답코드: {fast3r_response.status_code}")
                if fast3r_response.status_code == 200:
                    result = fast3r_response.json()
                    fast3r_responses.append({"file": os.path.basename(file_path), "result": result})
                else:
                    fast3r_responses.append({"file": os.path.basename(file_path),
                                             "error": f"HTTP {fast3r_response.status_code}"})
            except Exception as e:
                print(f"[❌] FAST3R 전송 실패: {e}")
                fast3r_responses.append({"file": os.path.basename(file_path), "error": str(e)})

    response_data = {
        "status": "success",
        "message": "이미지 저장 및 FAST3R 서버 전송 완료!",
        "fast3r_responses": fast3r_responses
    }

    response = make_response(jsonify(response_data))
    response.headers['Content-Type'] = 'application/json'
    return response

#2d_server resp,respone
@app.route('/2d_upload', methods=['POST'])
def request_2d_server():
    try:
        print("🔔 2D 서버로 요청 시작!")
        response_2d = requests.post(TWOD_SERVER_URL + "/2d_upload", timeout=300)

        if response_2d.status_code == 200:
            print("✅ 2D 서버 처리 완료:", response_2d.json())

            # ⭐️ 2D 처리 완료 후, FAST3R 서버에도 요청!
            print("🔔 FAST3R 서버로 이어서 요청 시작!")
            response_3d = requests.post(FAST3R_SERVER_URL + "/3d_upload", timeout=300)

            if response_3d.status_code == 200:
                print("✅ FAST3R 서버 처리 완료:", response_3d.json())
                return jsonify({
                    "status": "success",
                    "message": "2D 및 3D 처리 완료!",
                    "2d_result": response_2d.json(),
                    "3d_result": response_3d.json()
                })
            else:
                print("❌ FAST3R 서버 오류:", response_3d.text)
                return jsonify({
                    "status": "error",
                    "message": "3D 서버 오류: " + response_3d.text
                }), 500
        else:
            print("❌ 2D 서버 응답 오류:", response_2d.text)
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
