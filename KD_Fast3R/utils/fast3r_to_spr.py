import numpy as np
import pymeshlab

def postprocess(xyz, vertices, colors):

    """
    뜬금없이 지붕이 생기거나 원반형같이 나오는 등의 이상한 포인트 클라우드를 후처리하는 함수 (입출력: numpy)

    Args:
        xyz (np.ndarray): (N, 3) 원본 포인트 클라우드 정점 위치 좌표
        vertices (np.ndarray): (N, 3) 생성된 포인트 클라우드 정점 위치 좌표
        depth (np.ndarray): (N, 3) 생성된 포인트 클라우드 정점 색상 좌표

    Returns:
        vertices (np.ndarray): (N, 3) 후처리된 포인트 클라우드 정점 위치 좌표
        colors (np.ndarray): (N, 3) 후처리된 포인트 클라우드 정점 색상 좌표
    """

    # SPR 입력 범위 저장
    x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0]) # x 범위 설정
    y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1]) # y 범위 설정
    z_min, z_max = np.min(xyz[:, 2]), np.max(xyz[:, 2]) # z 범위 설정

    # 범위를 벗어난 포인트 제거
    in_bounds_mask = ( # 출력 정점이 입력 포인트의 범위 내에 있는지 확인하는 마스크
        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
        (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
    )
    vertices = vertices[in_bounds_mask] # 범위 안의 정점만 살림
    colors = colors[in_bounds_mask] # 범위 안의 색만 살림

    return vertices, colors
    
def spr(xyz, rgb, depth=9):

    """
    SPR 기반 메시 재구성 함수 (입출력: numpy)

    Args:
        xyz (np.ndarray): (N, 3) 원본 포인트 클라우드 정점 좌표
        rgb (np.ndarray): (N, 3) RGB 색상, [0~1] 범위
        depth (int): Poisson reconstruction 깊이

    Returns:
        vertices (np.ndarray): (N, 3) 메시 정점 좌표
        colors (np.ndarray): (N, 3) 메시 정점 색상 [0~1]
    """

    coords = xyz # xyz(N,3)
    colors = rgb # rgb(N,3)

    ms = pymeshlab.MeshSet()
    colors_4 = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)  # N,3 -> N,4

    m = pymeshlab.Mesh(vertex_matrix=coords, v_color_matrix=colors_4) # 좌표와 색상만으로 메쉬 객체 생성

    ms.add_mesh(m)
    ms.apply_filter('compute_normal_for_point_clouds') # 법선 벡터 자동 계산
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=depth) # 포아송 재구성 사용

    vertices = ms.current_mesh().vertex_matrix() # 정점(포인트 클라우드)
    colors = ms.current_mesh().vertex_color_matrix()[:, :3]  # RGBA → RGB
    
    vertices, colors = postprocess(xyz, vertices, colors) # 후처리

    return vertices, colors
