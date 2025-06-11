from server.server_maker import ServerMaker
from kd_fast3r.kd_fast3r import ShowRoom
from viz import ViserMaker

def set_2d_server(
        token='2xwkthyPz15CsSbartjgnt9aQde_3RoEvuB7Mz7oHHzuDJFia',
        url_type='TWOD_SERVER_URL',
        json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json',
        st_room_net_path='/content/drive/MyDrive/Final_Server/2d_server/ST_RoomNet',
        sd_path='/content/drive/MyDrive/Final_Server/2d_server/'
):
    s2 = ServerMaker(token=token, url_type=url_type, json_path=json_path)

    s2.set_2d(st_room_net_path, sd_path)

    s2.run()

def set_3d_server(
        token_3d='2y7j0XIpN1f2A76HJseTWFBqgqI_7Ma2wrEyqoveyw2JMeP6G',
        token_viser='2xwLYd6T4TFcrLjwgJZpxbDgaOJ_7oCZnw8f4Bkx2sYX3zkGQ',
        url_type='FAST3R_SERVER_URL',
        json_path='/content/drive/MyDrive/Final_Server/ngrok_path.json',
        model_path='jedyang97/Fast3R_ViT_Large_512',
        img_path='/content/drive/MyDrive/Final_Server/Input/Images',
        camera_path='/content/drive/MyDrive/Final_Server/Input/Poses/poses.txt',
        data_path='/content/drive/MyDrive/Final_Server/Input/Pts/fast3r_output.npz',
        info=True
):
    viz = ViserMaker(token_viser)

    sr = ShowRoom(model_path=model_path, img_path=img_path, camera_path=camera_path, data_path=data_path, info=info, viser=viz)

    s3 = ServerMaker(token=token_3d, url_type=url_type, json_path=json_path)

    s3.set_3d(sr)
    s3.set_viser(viz)
    s3.set_3d()

    s3.run()