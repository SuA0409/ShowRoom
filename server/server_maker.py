import json
from flask import Flask, jsonify, request, make_response
from pyngrok import ngrok
from flask_cors import CORS
from viz import find_free_port
from generate2d.discriminator.discriminator2d import dis_main
from generate2d.generator.stable_diffusion import init_set, gen_main
from kd_fast3r.utils.data_preprocess import server_images_load
from io import BytesIO
import requests
import copy
import base64

class ServerMaker:
    def __init__(self,
                 token=None,
                 url_type=None,
                 json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json',
                 ):
        self.token = token
        self.url_type = url_type
        self.json_path = json_path

        self._create_ngrok_flask_app()

    def _url_loader(self,
                    json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json'
                    ):
        with open(json_path, 'r') as f:
            url = json.load(f)

        self.REVIEW_SERVER_URL = url.get("REVIEW_SERVER_URL") or ""
        self.FAST3R_SERVER_URL = url.get("FAST3R_SERVER_URL") or ""
        self.TWOD_SERVER_URL = url.get("TWOD_SERVER_URL") or ""

        print(f"✅ REVIEW_SERVER_URL: {self.REVIEW_SERVER_URL}")
        print(f"✅ FAST3R_SERVER_URL: {self.FAST3R_SERVER_URL}")
        print(f"✅ TWOD_SERVER_URL: {self.TWOD_SERVER_URL}")

    def _url_saver(self,
                  public_url=None,
                  url_type=None,
                  json_path=None
                  ):

        assert public_url is not None, 'URL is not exist'

        try:
            with open(json_path, 'r') as f:
                url = json.load(f)
        except FileNotFoundError:
            url = dict()

        url[url_type] = public_url

        with open(json_path, 'w') as f:
            json.dump(url, f)

    def _create_ngrok_flask_app(self):
        assert self.token is not None, 'Token is not exist'

        # ngrok 토큰
        ngrok.set_auth_token(self.token)

        # Flask 초기화
        app = Flask(__name__)
        CORS(app)

        # ngrok 연결
        port = find_free_port()
        public_url = ngrok.connect(port).public_url
        print(f" {self.url_type} ngrok URL: {public_url}")

        if self.url_type is not None:
            self._url_saver(public_url=public_url, url_type=self.url_type, json_path=self.json_path)
        else:
            self._url_loader(json_path=self.json_path)

        self.app = app
        self.port = port

    def run(self):
        self.app.run(host='0.0.0.0', port=self.port)

    def set_3d(self, showroom):
        self.showroom = showroom

        @self.app.route('/3d_upload', methods=['POST'])
        def echo():
            try:
                print(' main에서 입력 받음 !')
                try:
                    self.showroom.images = server_images_load(request.files)
                except:
                    self.showroom.images = None

                self.showroom.reconstruction()
                self.showroom.building_spr()

                return jsonify({"pose": self.showroom.pose})
            except Exception as e:
                return f"Error processing: {str(e)}", 500

    def set_viser(self, show_viz):
        self.show_viz = show_viz

        @self.app.route('/viser', methods=['POST'])
        def viser_route():
            try:
                return jsonify({"status": str(self.show_viz.ngrok_url)})
            except Exception as e:
                return jsonify({"status": "fail", "error": str(e)})

    def set_2d(self):
        init_set()

        @self.app.route('/2d_upload', methods=['POST'])
        def handle_2d_request():
            try:
                print("    discriminator 실행 시작!")
                pose_json = request.form.get("pose")
                pose = json.loads(pose_json)
                result = dis_main(request.files, pose)

                print("    discriminator 실행 완료!")

                print("    Stable Diffusion Inpaint .py 실행 시작!")
                result = gen_main(result)

                print("    Stable Diffusion Inpaint.py 실행 완료!")

                encoded_images = []
                for name, bytesio_obj in result:
                    bytesio_obj.seek(0)
                    image_bytes = bytesio_obj.getvalue()
                    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                    encoded_images.append({
                        'name': name,
                        'data': encoded_image,
                        'format': 'base64'
                    })

                return jsonify({
                    "status": "success",
                    "message": "2D 생성 완료!",
                    "images": encoded_images
                })

            except Exception as e:
                print(" 2D 생성 중 오류:", e)
                return jsonify({"status": "error", "message": str(e)}), 500

    def set_main_3d(self):
    # 3d_server resp,respone
        @self.app.route('/3d_upload', methods=['POST'])
        def main_3d_process():
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
            self.files = []
            for i, url in enumerate(images):
                try:
                    response = requests.get(url)
                    response.raise_for_status()

                    # 메모리 상의 파일 객체 생성
                    img_file = BytesIO(response.content)
                    img_file.name = f'{i}.jpg'

                    # 원하는 형식으로 추가: (key, file-object)
                    self.files.append((f'images{i}', img_file))

                except Exception as e:
                    print(f"Failed to load image from {url}: {e}")

            print(self.files)
            # ✅ Fast3R 서버에 "폴더 전체 처리" 요청
            try:
                print("[⚡️] Fast3R에 요청 전송!")
                self.fast3r_response = requests.post(self.FAST3R_SERVER_URL + "/3d_upload", files=copy.deepcopy(self.files), timeout=600)
                print(f"[⚡️] Fast3R 응답코드: {self.fast3r_response.status_code}")

                if self.fast3r_response.status_code == 200:
                    fast3r_result = self.fast3r_response.json()
                else:
                    return jsonify({"status": "error", "message": f"Fast3R 오류: {self.fast3r_response.status_code}"}), 500
            except Exception as e:
                print(f"[❌] Fast3R 요청 실패: {e}")
                return jsonify({"status": "error", "message": f"Fast3R 요청 실패: {e}"}), 500

            print("viser 시작")

            # ✅ Viser에 "시각화 요청" 전송
            try:
                print("[⚡️] Viser에 요청 전송!")
                viser_response = requests.post(self.FAST3R_SERVER_URL + "/viser", timeout=600)
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
    def set_main_2d(self):

        @self.app.route('/2d_upload', methods=['POST'])
        def request_2d_server():
            print("🔔 2D 서버로 요청 시작!")
            try:
                data = {"pose": json.dumps(self.fast3r_response.json())}

                response_2d = requests.post(self.TWOD_SERVER_URL + "/2d_upload", files=copy.deepcopy(self.files),
                                            data=data,
                                            timeout=600)
                # dict(list[dict[file]])

                bytesio_obj = response_2d.json()['images'][0]['data']
                image_bytes = base64.b64decode(bytesio_obj)
                ## 체크
                print(image_bytes)
                bytesio_obj = BytesIO(image_bytes)
                print(bytesio_obj)
                name = response_2d.json()['images'][0]['name']
                # bytesio_obj.seek(0)

                print(f"{name}: {len(image_bytes)} bytes")
                new_files = copy.deepcopy(self.files)

                print(new_files)
                new_files.append((f'new_{name}', bytesio_obj))

                if response_2d.status_code == 200:
                    result_2d = response_2d.json()
                    print("✅ 2D 서버 처리 완료:", result_2d)

                    # ⭐️ 이어서 FAST3R 서버에 요청
                    print("🔔 FAST3R 서버로 요청 시작!")
                    response_3d = requests.post(self.FAST3R_SERVER_URL + "/3d_upload", files=copy.deepcopy(new_files),
                                                timeout=600)

                    if response_3d.status_code == 200:
                        result_3d = response_3d.json()
                        print("✅ FAST3R 처리 완료:", result_3d)

                        # ⭐️ ⭐️ 이어서 VISER 요청 추가!
                        print("🔔 VISER에 요청 시작!")
                        response_viser = requests.post(self.FAST3R_SERVER_URL + "/viser", timeout=600)

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
