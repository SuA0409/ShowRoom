from server.server_maker import ServerMaker
from kd_fast3r.kd_fast3r import ShowRoom
from viz import ViserMaker

def set_2d_server(
        token='2xwkthyPz15CsSbartjgnt9aQde_3RoEvuB7Mz7oHHzuDJFia',
        json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json'
):
    ''' 2d generate2d 를 운용하는 서버 생성
    Args:
        token (str): 2d 서버에 할당하는 ngrok 서버 토큰
        json_path (str): url을 메인에게 전달하기 위해 저정하는 주소
    '''

    # 서버를 생성하는 코드
    server2d = ServerMaker(token=token, url_type='TWOD_SERVER_URL', json_path=json_path)

    # 기본 설정과 라우터를 등록하는 코드
    server2d.set_2d()

    # 서버를 시작하는 코드
    server2d.run()

def set_3d_server(
        token_3d='2y7j0XIpN1f2A76HJseTWFBqgqI_7Ma2wrEyqoveyw2JMeP6G',
        token_viser='2xwLYd6T4TFcrLjwgJZpxbDgaOJ_7oCZnw8f4Bkx2sYX3zkGQ',
        json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json',
        info=True
):
    ''' 3d showroom을 운용하는 서버 생성
    Args:
        token_3d (str): 3d 서버에 할당하는 ngrok 서버 토큰
        token_viser (str): viser 서버에 할당하는 ngrok 서버 토큰
        json_path (str): url을 메인에게 전달하기 위해 저정하는 주소
        info (bool): fast3r에 정보 표시를 컨트롤하는 변수
    '''

    show_viz = ViserMaker(token_viser)
    showroom = ShowRoom(info=info, viz=show_viz)
    # 서버를 생성하는 코드
    server3d = ServerMaker(token=token_3d, url_type='FAST3R_SERVER_URL', json_path=json_path)

    # 기본 설정과 라우터를 등록하는 코드
    server3d.set_3d(showroom)

    # 서버를 시작하는 코드
    server3d.run()

def set_review(
        token = '2yGSKnM6Tviku0bqCV7bRN5y7gn_rLmTrz5SsPvRgd62yS5b',
        json_path = '/content/drive/MyDrive/Final_Server/ngrok_path.json'
):
    server_review = ServerMaker(token=token, url_type='REVIEW_SERVER_URL', json_path=json_path)

    # 기본 설정과 라우터를 등록하는 코드
    server_review.set_review()

    # 서버를 시작하는 코드
    server_review.run()

def set_main_server(
        token='2yRnnTsH8Hqdv2IJjnSCPnVOMVp_2E9gZxNeEhf3TfYPwKYnV',
        json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json',
):
    ''' 3d showroom을 운용하는 서버 생성
        Args:
            token (str): 3d 서버에 할당하는 ngrok 서버 토큰
            json_path (str): url을 메인에게 전달하기 위해 저정하는 주소
        '''

    # 서버를 생성하는 코드, url_type을 None으로 설정해 main이라는 것을 알림
    server = ServerMaker(token=token, url_type=None, json_path=json_path)

    # 기본 설정과 라우터를 등록하는 코드
    server.set_main()

    # 서버를 시작하는 코드
    server.run()