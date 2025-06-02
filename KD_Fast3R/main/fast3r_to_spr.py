import numpy as np
import pymeshlab

def spr(coords_np_Vx3, colors_np_Vx3, gt_normals_np_Vx3=None,
                                      depth=8, simplify_face_num=None, 
                                      xyz_bounds=None):
    """
    SPR 기반 메시 재구성 함수 (입출력: numpy), XYZ 좌표 범위 제한 기능 추가
    
    Args:
        coords_np_Vx3 (np.ndarray): (N, 3) 포인트 클라우드 정점 좌표
        colors_np_Vx3 (np.ndarray): (N, 3) RGB 색상, [0~1] 범위
        gt_normals_np_Vx3 (np.ndarray or None): (N, 3) 정규벡터, 없으면 자동 추정
        depth (int): Poisson reconstruction 깊이
        simplify_face_num (int or None): 지정하면 메시 간소화 수행
        xyz_bounds (tuple or None): ((x_min, x_max), (y_min, y_max), (z_min, z_max)) 범위 제한
                                   None이면 제한 없음

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

    # 현재 메시 가져오기
    vertices = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix()
    colors = ms.current_mesh().vertex_color_matrix()[:, :3]  # RGBA → RGB

    # XYZ 좌표 범위 제한
    if xyz_bounds is not None:
        x_bounds, y_bounds, z_bounds = xyz_bounds
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        z_min, z_max = z_bounds

        # 정점 필터링: XYZ 범위 내에 있는 정점만 유지
        valid_mask = (
            (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
            (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
            (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
        )
        valid_vertices_idx = np.where(valid_mask)[0]

        # 유효한 정점만 선택
        vertices = vertices[valid_vertices_idx]
        colors = colors[valid_vertices_idx]

        # 얼굴 재인덱싱: 유효한 정점 인덱스에 맞게 얼굴 업데이트
        valid_faces_mask = np.all(np.isin(faces, valid_vertices_idx), axis=1)
        faces = faces[valid_faces_mask]

        # 정점 인덱스 재매핑
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertices_idx)}
        faces = np.array([[index_map.get(idx, -1) for idx in face] for face in faces])
        faces = faces[np.all(faces != -1, axis=1)]  # 유효하지 않은 얼굴 제거

    return vertices, colors
