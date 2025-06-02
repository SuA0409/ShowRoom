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
    coords = coords_np_Vx3
    colors = colors_np_Vx3
    normals = gt_normals_np_Vx3
    use_GT_normals = gt_normals_np_Vx3 is not None

    ms = pymeshlab.MeshSet()
    colors_4 = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)  # N,3 -> N,4

    if use_GT_normals:
        m = pymeshlab.Mesh(vertex_matrix=coords, v_normals_matrix=normals, v_color_matrix=colors_4)
    else:
        m = pymeshlab.Mesh(vertex_matrix=coords, v_color_matrix=colors_4)

    ms.add_mesh(m)

    if not use_GT_normals:
        ms.apply_filter('compute_normal_for_point_clouds')

    ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=depth)

    if simplify_face_num is not None:
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=simplify_face_num, preservetopology=True)

    vertices = ms.current_mesh().vertex_matrix()
    colors = ms.current_mesh().vertex_color_matrix()[:, :3]  # RGBA → RGB

    return vertices, colors

def postprocess(xyz, vertices, colors):
    # SPR 입력 범위 저장
    x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    z_min, z_max = np.min(xyz[:, 2]), np.max(xyz[:, 2])

    # 범위를 벗어난 포인트 제거
    in_bounds_mask = (
        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
        (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
    )
    vertices = vertices[in_bounds_mask]
    colors = colors[in_bounds_mask]
    return vertices, colors
