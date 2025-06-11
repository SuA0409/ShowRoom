import json
from flask import Flask, jsonify, request, render_template, make_response
from pyngrok import ngrok
from flask_cors import CORS
from viz import find_free_port
import time
from ST_RoomNet.ST_RoomNet import ShowRoomProcessor
from StableDiffusion.StableDiffusionInpaint import main as sd_main

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
        assert self.url_type in ['REVIEW_SERVER_URL', 'FAST3R_SERVER_URL', 'TWOD_SERVER_URL'], 'URL type is not defined'

        # ngrok 토큰
        ngrok.set_auth_token(self.token)

        # Flask 초기화
        app = Flask(__name__)
        CORS(app)

        # ngrok 연결
        port = find_free_port()
        public_url = ngrok.connect(port).public_url
        print(f" {self.url_type} ngrok URL: {public_url}")

        self._url_saver(public_url=public_url, url_type=self.url_type, json_path=self.json_path)

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
                time.sleep(1)
                self.showroom.reconstruction()

                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": str(e)})


        @self.app.route('/spr', methods=['POST'])
        def spr():
            try:
                self.showroom.building_spr()

                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": str(e)})

    def set_viser(self, show_viz):
        self.show_viz = show_viz

        @self.app.route('/viser', methods=['POST'])
        def viser_route():
            try:
                return jsonify({"status": str(self.show_viz.ngrok_url)})
            except Exception as e:
                return jsonify({"status": "fail", "error": str(e)})

    def set_2d(self,
               st_room_net_path='/content/drive/MyDrive/Final_Server/2d_server/ST_RoomNet',
               sd_path='/content/drive/MyDrive/Final_Server/2d_server/'
               ):
        @self.app.route('/2d_upload', methods=['POST'])
        def handle_2d_request():
            try:
                print("    ST_RoomNet 실행 시작!")
                processor = ShowRoomProcessor()
                processor.process()

                print("    ST_RoomNet 실행 완료!")

                print("    Stable Diffusion Inpaint .py 실행 시작!")
                sd_main()
                print("    Stable Diffusion Inpaint.py 실행 완료!")

                return jsonify({"status": "success", "message": "2D 생성 완료!"})

            except Exception as e:
                print(" 2D 생성 중 오류:", e)
                return jsonify({"status": "error", "message": str(e)}), 500