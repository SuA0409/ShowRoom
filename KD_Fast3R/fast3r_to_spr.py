import numpy as np
import pymeshlab

def spr(coords_np_Vx3, colors_np_Vx3, gt_normals_np_Vx3=None,
                                      depth=8, simplify_face_num=None):
    """
    SPR 기반 메시 재구성 함수 (입출력: numpy)

    Args:
        coords_np_Vx3 (np.ndarray): (N, 3) 포인트 클라우드 정점 좌표
        colors_np_Vx3 (np.ndarray): (N, 3) RGB 색상, [0~1] 범위
        gt_normals_np_Vx3 (np.ndarray or None): (N, 3) 정규벡터, 없으면 자동 추정
        depth (int): Poisson reconstruction 깊이
        simplify_face_num (int or None): 지정하면 메시 간소화 수행

    Returns:
        vertices (np.ndarray): (M, 3) 메시 정점 좌표
        faces (np.ndarray): (K, 3) 메시 삼각형 인덱스
        colors (np.ndarray): (M, 3) 메시 정점 색상 [0~1]
    """
    coords = coords_np_Vx3 # xyz(N,3)
    colors = colors_np_Vx3 # rgb(N,3)
    normals = gt_normals_np_Vx3 # 입력 정규 벡터
    use_GT_normals = gt_normals_np_Vx3 is not None # True -> 사용, False -> 사용X

    ms = pymeshlab.MeshSet()
    colors_4 = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)  # N,3 -> N,4

    if use_GT_normals:
        m = pymeshlab.Mesh(vertex_matrix=coords, v_normals_matrix=normals, v_color_matrix=colors_4) # 좌표, 정규벡터, 색상으로 메쉬 생성
    else:
        m = pymeshlab.Mesh(vertex_matrix=coords, v_color_matrix=colors_4) # 좌표와 색상만으로 메쉬 객체 생성

    ms.add_mesh(m)

    if not use_GT_normals:
        ms.apply_filter('compute_normal_for_point_clouds') # 법선 벡터 자동 계산

    ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=depth) # 포아송 재구성 사용

    if simplify_face_num is not None: # 메쉬 간소화
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=simplify_face_num, preservetopology=True)

    vertices = ms.current_mesh().vertex_matrix() # 정점(포인트 클라우드)
    colors = ms.current_mesh().vertex_color_matrix()[:, :3]  # RGBA → RGB

    return vertices, colors

# 뜬금없이 지붕이 생기거나 원반형같이 나오는 등의 이상한 포인트 클라우드를 후처리
def postprocess(xyz, vertices, colors):
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