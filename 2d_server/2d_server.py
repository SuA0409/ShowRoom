import sys
sys.path.append('/content/drive/MyDrive/Final_Server')
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from pyngrok import ngrok
import os, requests, json, subprocess, re
from utils import create_ngrok_flask_app

app = create_ngrok_flask_app(token="2xwkthyPz15CsSbartjgnt9aQde_3RoEvuB7Mz7oHHzuDJFia", url_type='TWOD_SERVER_URL')

@app.route('/2d_upload', methods=['POST'])
def handle_2d_request():
    try:
        print("🔔 ST-RoomNet 실행 시작!")
        subprocess.run(
    ["python", "ST_RoomNet.py"],
    cwd="/content/drive/MyDrive/Final_Server/2d_server/ST-RoomNet",
    check=True
)

        print("✅ ST-RoomNet 실행 완료!")

        print("🔔 rotate_and_inpainting.py 실행 시작!")
        subprocess.run(
    ["python", "rotate_and_inpaint.py"],
    cwd="/content/drive/MyDrive/Final_Server/2d_server/",
    check=True
)
        print("✅ rotate_and_inpainting.py 실행 완료!")

        return jsonify({"status": "success", "message": "2D 생성 완료!"})

    except subprocess.CalledProcessError as e:
        print("❌ 2D 생성 중 오류:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
