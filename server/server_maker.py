import json
from flask import Flask, jsonify, request, make_response, render_template
from pyngrok import ngrok
from flask_cors import CORS
from io import BytesIO
import requests
import copy
import base64
import os
import re

from viz.viz import find_free_port
from generate2d.discriminator.discriminator2d import dis_main
from generate2d.generator.stable_diffusion import init_set, gen_main
from kd_fast3r.utils.data_preprocess import server_images_load
from review.main_review import get_reviews, preprocess_reviews, use_model, get_review_conf

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
                    public_url="",
                    json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json'
                    ):
        with open(json_path, 'r') as f:
            url = json.load(f)

        # main server에서 저장된 각각의 server 주소 load
        self.REVIEW_SERVER_URL = url.get("REVIEW_SERVER_URL") or ""
        self.FAST3R_SERVER_URL = url.get("FAST3R_SERVER_URL") or ""
        self.TWOD_SERVER_URL = url.get("TWOD_SERVER_URL") or ""
        self.MAIN_SERVER_URL = public_url

        print(f"    1. REVIEW_SERVER_URL: {self.REVIEW_SERVER_URL}")
        print(f"    2. FAST3R_SERVER_URL: {self.FAST3R_SERVER_URL}")
        print(f"    3. TWOD_SERVER_URL: {self.TWOD_SERVER_URL}")

    def _url_saver(self,
                  public_url=None,
                  url_type=None,
                  json_path=None
                  ):
        ''' url을 저장하는 코드
        Args:
            public_url (str): ngrok으로 생성된 주소
            url_type (str): josn에 저장할 url 이름
            json_path (str): url이 저장된 json 파일의 주소
        '''

        # 주소값이 없으면 실행 불가
        assert public_url is not None, 'URL is not exist'

        try:
            with open(json_path, 'r') as f:
                url = json.load(f)
        except FileNotFoundError:
            # json 파일이 없으면 dictionary를 다시 만듦
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

        if self.url_type == "MAIN_SERVER_URL":
            self._url_loader(public_url=public_url, json_path=self.json_path)
        else:
            self._url_saver(public_url=public_url, url_type=self.url_type, json_path=self.json_path)

        self.app = app
        self.port = port

    def run(self):
        # 서버 실행
        self.app.run(host='0.0.0.0', port=self.port)

    def set_3d(self, showroom):

        # 3d_upload에 관한 라우터
        @self.app.route('/3d_upload', methods=['POST'])
        def show3r_route():
            try:
                print(' main에서 입력 받음 !')

                # request로 입력받은 데이터를 showroom.room에 저장
                showroom.room = server_images_load(request.files)

                # fast3r 실행
                showroom.reconstruction()
                # spr 실행
                showroom.building_spr()

                # camera pose를 return
                return jsonify({"pose": showroom.pose})
            except Exception as e:
                return f"Error processing: {str(e)}", 500

        # viser에 관한 라우터
        @self.app.route('/viser', methods=['POST'])
        def viser_route():
            try:
                # viser 주소를 return
                return jsonify({"status": str(showroom.viz.ngrok_url)})
            except Exception as e:
                return jsonify({"status": "fail", "error": str(e)})

    def set_2d(self):

        # generator(stable diffusion)에 필요한 기본 setting
        init_set()

        # 2d_upload에 관한 라우터
        @self.app.route('/2d_upload', methods=['POST'])
        def show_gen_route():
            print(' main에서 입력 받음 !')
            try:
                ## dis 파트
                print(" discriminator 실행 시작!")

                # 입력받은 camere pose를 처리
                pose_json = request.form.get("pose")
                pose = json.loads(pose_json)
                dis_result = dis_main(request.files, pose)

                print(" discriminator 실행 완료!\n")

                if dis_result is None:
                    return jsonify({"status": "error", "message": '생성할 이미지 없음'}), 500

                ## gen 파트
                print(" Stable Diffusion Inpaint .py 실행 시작!")
                gen_result = gen_main(dis_result)

                print(" Stable Diffusion Inpaint.py 실행 완료!")

                encoded_images = []
                for name, bytesio_obj in gen_result:
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

    def set_review(self):
        RESULT_FOLDER = 'server/results'
        os.makedirs(RESULT_FOLDER, exist_ok=True)

        # 리뷰 분석만 수행하는 엔드포인트
        @self.app.route('/analyze_review', methods=['POST'])
        def analyze_review():
            print(' main에서 입력 받음 !')

            try:
                data = request.get_json()
                url = data.get('url')
                print(" 리뷰 분석 요청 받은 URL:", url)

                if url is None:
                    return jsonify({"status": "error", "message": "숙소 URL이 없습니다."}), 400

                # 숙소 ID 추출 (파일명으로도 사용)
                room_id_match = re.search(r'/rooms/(\d+)', url)
                if not room_id_match:
                    return jsonify({"status": "error", "message": "유효한 숙소 URL이 아닙니다."}), 400
                room_id = room_id_match.group(1)

                # 리뷰 분석 실행
                review_conf = get_review_conf()
                data, num = get_reviews(url, review_conf['headers'])
                docs = preprocess_reviews(data, num)
                topic_sentences = use_model(docs, review_conf['seed_topics'])

                result = {"stay_id": num, "topics": topic_sentences}

                # 결과를 JSON 파일로 저장 (선택적)
                save_path = os.path.join(RESULT_FOLDER, f"review.json")
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)

                # 최종 결과 JSON으로 반환
                return jsonify({
                    "status": "success",
                    "room_id": room_id,
                    "result": result
                })

            except Exception as e:
                print(f" 리뷰 분석 실패: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

    def set_main(self):
        RESULTS_FOLDER = 'server/results'

        @self.app.route('/analyze_review', methods=['POST'])
        def analyze_review():
            data = request.get_json()
            print(f" analyze_review 요청 데이터: {data}")

            if not data or 'url' not in data:
                return jsonify({"status": "error", "message": "요청에 URL이 없습니다."}), 400

            url = data['url']
            print(f" 리뷰 분석 요청된 URL: {url}")

            try:
                headers = {'Content-Type': 'application/json'}
                print(" 리뷰 분석 서버에 요청 전송 !")
                review_response = requests.post(self.REVIEW_SERVER_URL + "/analyze_review",
                                                json={"url": url}, headers=headers, timeout=300)

                print(f" 리뷰 분석 서버 응답 코드: {review_response.status_code}\n")

                if review_response.status_code == 200:
                    result = review_response.json()
                else:
                    print(f" 리뷰 분석 서버 응답 오류: {review_response.status_code}")
                    return jsonify({"status": "error", "message": f"리뷰 분석 서버 오류: {review_response.status_code}"}), 500

                if result.get("status") == "success":
                    room_id = result.get('room_id', 'unknown')

                    json_path = os.path.join(RESULTS_FOLDER, f"{room_id}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result['result'], f, ensure_ascii=False, indent=2)
                    print(f"    결과 JSON 저장 완료: {json_path}")

                    view_url = f"{self.MAIN_SERVER_URL}/review/{room_id}"

                    return jsonify({"status": "success", "view_url": view_url})
                else:
                    error_msg = result.get("message", "Unknown error")
                    print(f" 리뷰 분석 실패: {error_msg}")
                    return jsonify({"status": "error", "message": "리뷰 분석 실패: " + error_msg}), 500

            except Exception as e:
                print(f" 리뷰 분석 서버 요청 실패: {e}")
                return jsonify({"status": "error", "message": f"리뷰 분석 서버 통신 실패! {e}"}), 500

        @self.app.route('/review/<room_id>')
        def show_review(room_id):
            print(f" {room_id} 호출됨")
            json_path = os.path.join(RESULTS_FOLDER, f"{room_id}.json")
            if not os.path.exists(json_path):
                print(f" 결과 JSON 파일이 없음: {json_path}")
                return "분석 결과가 없습니다.", 404

            with open(json_path, encoding='utf-8') as f:
                result = json.load(f)

            return render_template("review_result.html", room_id=room_id, topics=result['topics'])

        # 3d_server resp,respone
        @self.app.route('/3d_upload', methods=['POST'])
        def main_3d_process():

            # chrome에서 전달 받은 이미지 url
            data = request.get_json()
            print(f" Fast3r 요청 데이터: {data}")

            # 입력 검증
            if not data or 'images' not in data:
                return jsonify({"status": "error", "message": "요청에 이미지 정보가 없습니다."}), 400

            images = data['images']
            if len(images) < 1:
                return jsonify({"status": "error", "message": "저장할 이미지가 없습니다."}), 400

            print(f"    저장할 이미지 리스트: {images}")

            ## 이미지 byte로 변환
            self.files = list()
            for i, url in enumerate(images):
                try:
                    response = requests.get(url)
                    response.raise_for_status()

                    # 메모리 상의 파일 객체 생성
                    img_file = BytesIO(response.content)
                    img_file.name = f'{i}.jpg'

                    # 이미지를 (name, Byte) 형식으로 저장
                    self.files.append((f'images{i}', img_file))

                except Exception as e:
                    print(f"Failed to load image from {url}: {e}")

            print(f"    저장한 이미지 리스트: {self.files}")

            ## Fast3r에 3d 전환 요청
            try:
                print(" Fast3R에 요청 전송 !")
                self.fast3r_response = requests.post(self.FAST3R_SERVER_URL + "/3d_upload", files=copy.deepcopy(self.files), timeout=600)

                if self.fast3r_response.status_code == 200:
                    fast3r_result = self.fast3r_response.json()
                    print(f" Fast3R 응답 코드: {self.fast3r_response.status_code}\n")
                else:
                    return jsonify({"status": "error", "message": f"Fast3R 오류: {self.fast3r_response.status_code}"}), 500

            except Exception as e:
                print(f" Fast3R 요청 실패: {e}\n")
                return jsonify({"status": "error", "message": f"Fast3R 요청 실패: {e}"}), 500

            ## Viser에 시각화 요청
            try:
                print(" Viser에 요청 전송 !")
                viser_response = requests.post(self.FAST3R_SERVER_URL + "/viser", timeout=600)

                if viser_response.status_code == 200:
                    viser_result = viser_response.json()
                    print(f" Viser 응답 코드: {viser_response.status_code}\n")
                else:
                    return jsonify({"status": "error", "message": f"Viser 오류: {viser_response.status_code}"}), 500
            except Exception as e:
                print(f" Viser 요청 실패: {e}\n")
                return jsonify({"status": "error", "message": f"Viser 요청 실패: {e}"}), 500

            ## 최종 응답 통합
            response_data = {
                "status": "success",
                "message": "이미지 저장, Fast3R 처리 및 Viser 요청까지 완료 !",
                "fast3r_response": fast3r_result,
                "viser_response": viser_result
            }

            response = make_response(jsonify(response_data))
            response.headers['Content-Type'] = 'application/json'
            return response

        # 2d_server resp,respone
        @self.app.route('/2d_upload', methods=['POST'])
        def request_2d_server():
            # 2d server에 생성 요청
            try:
                ## fast3r 결과로 얻은 camera pose를 전달
                data = {"pose": json.dumps(self.fast3r_response.json())}
                print(" 2D server에 요청 전송 !")
                response_2d = requests.post(self.TWOD_SERVER_URL + "/2d_upload",
                                            files=copy.deepcopy(self.files), data=data, timeout=600)

                if response_2d.status_code == 200:
                    result_2d = response_2d.json()
                    print(f" 2D server 응답 코드: {response_2d.status_code}\n")
                else:
                    return jsonify({"status": "error", "message": "2D 서버 오류: " + response_2d.text}), 500

                print(" Fast3R에 요청 준비 !")

                ## 2d 결과로 얻은 새로운 이미지를 byte로 압축 하여 main에 전달
                bytesio_obj = response_2d.json()['images'][0]['data']
                image_bytes = base64.b64decode(bytesio_obj)
                bytesio_obj = BytesIO(image_bytes)
                name = response_2d.json()['images'][0]['name']

                print(f"    생성된 이미지 이름: {name}")
                print(f"    생성된 이미지 크기: {len(image_bytes)} bytes")

                new_files = copy.deepcopy(self.files)
                new_files.append((f'new_{name}', bytesio_obj))
                print(f"    새롭게 저장한 이미지 리스트: {new_files}\n")

                ## fast3r에 두 번째 요청
                print(" Fast3R에 두 번째 요청 전송 !")
                response_3d = requests.post(self.FAST3R_SERVER_URL + "/3d_upload",
                                            files=copy.deepcopy(new_files), timeout=600)

                if response_3d.status_code == 200:
                    result_3d = response_3d.json()
                    print(f" Fast3R 응답 코드: {self.fast3r_response.status_code}\n")
                else:
                    return jsonify({"status": "error", "message": "3D 서버 오류: " + response_3d.text}), 500


                ## Viser에 두 번째 요청
                print(" Viser에 두 번째 요청 전송 !")
                response_viser = requests.post(self.FAST3R_SERVER_URL + "/viser", timeout=600)

                if response_viser.status_code == 200:
                    result_viser = response_viser.json()
                    print(f" Viser 응답 코드: {response_viser.status_code}\n")

                    print(" *** 전체 프로세스 종료 ***")
                    return jsonify({
                        "status": "success",
                        "message": "2D, 3D, Viser까지 모두 완료!",
                        "2d_result": result_2d,
                        "3d_result": result_3d,
                        "viser_result": result_viser
                    })
                else:
                    return jsonify({"status": "error", "message": f"Viser 오류: {response_viser.text}"}), 500

            except Exception as e:
                print(" 2D 서버 요청 실패:", e)
                return jsonify({"status": "error", "message": str(e)}), 500

