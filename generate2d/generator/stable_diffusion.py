import cv2
import os
import time
import torch
import numpy as np
import kornia as K
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from torch import autocast
from io import BytesIO

class SimpleRotator:
    """
    2D 이미지를 3D 공간에서 회전시킨 후 빈 영역을 자연스럽게 인페인팅하는 클래스
    주요 기능:
    1. MiDaS를 이용한 깊이 맵 예측
    2. 역투영 → Y축 회전 → 재투영
    3. 빈 영역 마스크 생성
    4. Stable Diffusion Inpainting으로 빈 영역 채우기
    """
    output = "/content/ShowRoom/demo/data" # Input Your Name of Image Folder

    
    def __init__(self, device='cuda', max_depth_m=5.0, depth_model='MiDaS_small'):
        '''
        클래스 초기화 함수, 필요한 모델들을 로드하고 초기 설정을 수행
        Args:
            device (str): 연산에 사용할 디바이스 ('cuda' 또는 'cpu')
            max_depth_m (float): 예측된 깊이 맵을 정규화할 때 사용할 최대 깊이 값
            depth_model (str): 깊이 예측에 사용할 MiDaS 모델의 이름.
        '''
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 깊이 정규화 최대값
        self.max_depth_m = max_depth_m

        # MiDaS 깊이 예측 모델 로딩
        self.midas = torch.hub.load(
            "intel-isl/MiDaS",
            depth_model,
            map_location=self.device,
            force_reload=False,
            trust_repo=True,
            offline=True  # 로컬 캐시에 모델이 없으면 에러 발생
        ).to(self.device).eval()
        try:
            self.midas = torch.compile(self.midas)
        except Exception:
            pass

        # MiDaS 모델에 맞는 입력 전처리기(transform)를 로드
        trans = torch.hub.load("intel-isl/MiDaS", "transforms")
        if depth_model == 'MiDaS_small' and hasattr(trans, 'small_transform'):
            self.transform = trans.small_transform
        else:
            self.transform = trans.dpt_transform

        # Stable Diffusion Inpainting 파이프라인 로딩
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True  # 로컬 캐시에 모델이 없으면 에러 발생
        ).to(self.device)

        # 모델 최적화 설정
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except ModuleNotFoundError:
            print("Warning: xFormers not installed, using default attention")

        # 적은 추론 스텝으로 안정적인 결과를 효율적으로 생성
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def estimate_depth(self, img_rgb: np.ndarray) -> torch.Tensor:
        '''
        MiDaS 모델을 사용하여 입력 이미지의 깊이 맵을 예측합니다.
        Args:
            img_rgb (np.ndarray): HxWx3 형태의 RGB 이미지
        Return:
            torch.Tensor: 1x1xHxW 형태의 깊이 맵 텐서. 값의 범위는 [0, max_depth_m]
        '''
        # 이미지의 높이와 너비 추출
        H, W = img_rgb.shape[:2]

        # MiDaS 모델의 전처리기를 적용하고 텐서를 지정된 디바이스로 이동
        inp = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            depth = self.midas(inp)

        # MiDaS 출력 텐서가 3차원일 경우, 채널 차원을 추가하여 4차원으로 만듬 (B, C, H, W)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        # 예측된 깊이 맵을 원본 이미지와 동일한 해상도로 리사이즈
        depth = torch.nn.functional.interpolate(
            depth, size=(H, W), mode='bicubic', align_corners=False
        )

        # 깊이 값을 [0, 1] 범위로 정규화
        dmin, dmax = depth.min(), depth.max()
        depth = (depth - dmin) / (dmax - dmin)
        return depth * self.max_depth_m

    @staticmethod
    def unproject(depth: torch.Tensor, Kmat: torch.Tensor) -> torch.Tensor:
        '''
        2D 깊이 맵과 카메라 내부 행렬(K)을 사용하여 각 픽셀을 3D 공간 좌표로 변환(역투영)합니다.
        Args:
            depth (torch.Tensor): 1x1xHxW 형태의 깊이 맵 텐서
            Kmat (torch.Tensor): 3x3 형태의 카메라 내부 행렬
        Return:
            torch.Tensor: 1xHxWx3 형태의 3D 포인트 클라우드 텐서
        '''
        _, _, H, W = depth.shape

        # 깊이 값(z) 추출
        zs = depth[0, 0]

        # 픽셀 좌표 그리드 생성 (indexing='ij'는 행렬 인덱싱 방식과 동일하게 y, x 순서)
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        xs, ys = xs.to(depth.device).float(), ys.to(depth.device).float()

        # 동차 좌표계로 변환 준비(x*z, y*z, z)
        xz = xs * zs
        yz = ys * zs
        xyz = torch.stack([xz, yz, zs], dim=-1)

        # 카메라 내부 행렬의 역행렬을 사용하여 3D 좌표로 변환
        Kinv = torch.inverse(Kmat)

        # 각 픽셀의 3D 좌표/값을 변환
        flat = xyz.view(-1, 3) @ Kinv.T

        return flat.view(1, H, W, 3)

    @staticmethod
    def project(xyz: torch.Tensor, Kmat: torch.Tensor) -> torch.Tensor:
        '''
        3D 공간 좌표를 2D 이미지 평면으로 투영하고, grid_sample 함수에 사용될 정규화된 좌표 그리드를 생성합니다.
        Args:
            xyz (torch.Tensor): 1xHxWx3 형태의 3D 포인트 클라우드 텐서
            Kmat (torch.Tensor): 3x3 형태의 카메라 내부 행렬
        Return:
            torch.Tensor: 1xHxWx2 형태의 정규화된 샘플링 그리드 텐서. 값의 범위는 [-1, 1]입니다.
        '''
        # 3D 포인트 클라우드의 차원 정보
        B, H, W, _ = xyz.shape

        # (B, H, W, 3) -> (B*H*W, 3)로 펼쳐서 행렬 곱셈 준비
        flat = xyz.view(-1, 3)

        # 3D 포인트를 카메라 내부 행렬과 곱하여 2D 이미지 평면으로 투영
        proj = (flat @ Kmat.T).view(B, H, W, 3)

        # 동차 좌표 분할(Perspective division)을 통해 2D 좌표 (x, y)를 얻음
        xy = proj[..., :2] / (proj[..., 2:3] + 1e-8)

        # grid_sample 함수를 위해 좌표를 [-1, 1] 범위로 정규화
        grid = torch.stack([
            2 * xy[..., 0] / (W - 1) - 1,
            2 * xy[..., 1] / (H - 1) - 1
        ], dim=-1)
        return grid

    def rotate_frame(self, img_rgb: np.ndarray, angle_deg: float):
        '''
        깊이 정보를 활용하여 2D 이미지를 3D 공간에서 회전시킨 후,
        회전된 이미지와 빈 영역을 나타내는 마스크를 생성
        Args:
            img_rgb (np.ndarray): HxWx3 형태의 원본 RGB 이미지
            angle_deg (float): Y축 기준 회전 각도 (degree) 양수는 시계 방향, 음수는 반시계 방향
        Return:
            new_rgb (torch.Tensor): 1x3xHxW 형태의 회전된 이미지 텐서 (값 범위: [-1, 1])
            depth (torch.Tensor): 1x1xHxW 형태의 예측된 깊이 맵 텐서
            mask (torch.Tensor): 1x1xHxW 형태의 빈 영역 마스크 텐서 (1: 빈 영역, 0: 유효 영역)
        '''
        # 입력 이미지를 텐서로 변환하고 [-1, 1] 범위로 정규화
        img = torch.from_numpy(img_rgb).permute(2, 0, 1)[None].float().to(self.device)
        img = img / 127.5 - 1

        # 깊이 맵 예측
        depth = self.estimate_depth(img_rgb).to(self.device)
        H, W = img_rgb.shape[:2]

        # 카메라 내부 행렬(K) 생성
        f = min(W, H) * 0.8
        Kmat = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=self.device)

        # 깊이 맵을 3D 포인트 클라우드로 역투영
        xyz = self.unproject(depth, Kmat)

        # Y축 기준 3D 회전 행렬 생성
        theta = angle_deg / 180 * np.pi
        axis = torch.tensor([0.0, -1.0, 0.0], device=self.device)
        rotmat = K.geometry.conversions.axis_angle_to_rotation_matrix((axis * theta)[None])

        # 3D 포인트 클라우드를 회전시키고, 다시 2D로 투영하여 샘플링 그리드 생성
        xyz_rot = torch.matmul(xyz, rotmat[0, :3, :3].T)
        grid = self.project(xyz_rot, Kmat)

        # grid_sample을 사용하여 원본 이미지로부터 회전된 이미지를 샘플링
        new_rgb = torch.nn.functional.grid_sample(
            img, grid, align_corners=False, mode='bilinear', padding_mode='zeros'
        )

        # 샘플링 그리드 좌표가 [-1, 1] 범위를 벗어나는 영역을 찾아 마스크 생성
        mask = ((grid[..., 0].abs() > 1) | (grid[..., 1].abs() > 1)).unsqueeze(0).float()
        return new_rgb, depth, mask

    def inpaint(self, init_image: Image.Image, mask_image: Image.Image,
                prompt: str, steps: int = 50, guidance: float = 8.5) -> Image.Image:
        '''
        Stable Diffusion Inpainting 파이프라인을 사용하여 이미지의 마스크된 영역을 채웁니다.
        Args:
            init_image (Image.Image): 회전되었지만 아직 채워지지 않은 초기 이미지 (PIL Image)
            mask_image (Image.Image): 채워야 할 영역을 나타내는 마스크 이미지 (PIL Image, L-mode)
            prompt (str): 인페인팅 과정에 대한 지침을 제공하는 텍스트 프롬프트
            steps (int): 인페인팅 추론(inference) 스텝 수
            guidance (float): 프롬프트의 영향을 조절하는 guidance scale 값
        Return:
            Image.Image: 인페인팅이 완료된 최종 이미지 (PIL Image)
        '''
        # 모델 파이프라인에 입력값 전달하여 최종 이미지 생성
        with autocast(self.device.type):
            result = self.pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                guidance_scale=guidance,
                num_inference_steps=steps
            ).images[0]
        return result

    def to_bytesio(self, image_np: np.ndarray, filename: str = "result.jpg") -> BytesIO:
        """
        NumPy 배열 형태의 이미지를 JPEG 형식으로 인코딩하여 BytesIO 객체로 반환합니다.
        Args:
            image_np(np.ndarray): H×W×C 형태의 이미지 배열 (uint8)
            filename(str): 생성될 BytesIO 객체의 파일 이름 (기본값: "result.jpg")
        Return:
            BytesIO:JPEG 인코딩된 이미지 데이터를 담은 BytesIO 객체
            반환된 객체의 name 속성에 filename이 설정되어 있습니다.
        """
        # NumPy 배열을 JPEG 형식으로 인코딩
        success, encoded_image = cv2.imencode('.jpg', image_np)
        # 인코딩이 실패했으면 예외를 발생시킵니다
        if not success:
            raise ValueError("JPEG 인코딩 실패")
        
        # 인코딩된 바이트 데이터를 BytesIO 객체로 래핑하여 파일 형태로 만듭니다
        img_file = BytesIO(encoded_image.tobytes())
        
        # 생성된 BytesIO 객체에 파일 이름 설정
        img_file.name = filename
        
        return img_file

def init_set():
    """
    MiDaS 및 Stable Diffusion Inpainting 모델을 로컬 캐시에 다운로드하고 초기 설정을 수행합니다.
    Args:
        없음
    Return:
        None: 다운로드가 완료되면 콘솔에 "다운로드 완료!" 메시지를 출력합니다.
    """
    # MiDaS_small 모델 다운로드 (offline=False 옵션으로 캐시에 없으면 원격에서 가져옴)
    torch.hub.load("intel-isl/MiDaS", "MiDaS_small", offline=False)

    # MiDaS 전처리용 transforms 다운로드 (offline 옵션 없이)
    torch.hub.load("intel-isl/MiDaS", "transforms")

    # Stable Diffusion Inpainting 다운로드
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=False
    )
    print("다운로드 완료!")

def show_image(img: Image.Image, title: str = "image", save_dir: str = "output"):
    """
    PIL 이미지를 시각화하는 대신 파일로 저장 (headless 환경 대응)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{title}.jpg")
    img.save(save_path)
    print(f" 저장 완료: {save_path}")

def gen_main(output_list):
    """
    주어진 이미지 리스트를 순회하며 회전 → 빈 공간 마스크 생성 → Stable Diffusion 인페인팅을
    통해 빈 영역을 채워 최종 이미지를 생성하고, BytesIO 객체로 반환 리스트를 구성합니다.
    Args:
        output_list (list of dict): 처리할 이미지 정보 리스트
            각 원소는 다음 키를 가져야 합니다:
            - 'key' (int): 0이면 시계 방향(+angle_value), 1이면 반시계 방향(-angle_value) 회전
            - 'image' (np.ndarray): H×W×3 RGB uint8 이미지 배열
    Return:
        list of (str, BytesIO): 
            처리된 결과 이미지 파일 리스트.
            각 튜플의 첫 요소는 파일 이름 접두사(f"images{key}")이며,
            두 번째 요소는 JPEG 인코딩된 BytesIO 객체입니다.
    """
    start_time = time.time()

    # 파라미터 값 수정
    seed = 42
    angle_value = 30
    steps = 50
    guidance = 8.5
    prompt = "Extend only the background wall and floor. Do not add new objects or decorations. Match color and lighting. Keep everything minimal."

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 회전 및 인페인팅 수행용 SimpleRotator 인스턴스 생성
    rotator = SimpleRotator(device='cuda', max_depth_m=3.0)

    # 결과 저장
    file = list()
    # output_list가 None이거나 비어 있으면 바로 반환
    if not output_list:
        raise ValueError("생성하기 적합하기 어려운 방임! 인져어어엉? by 수한") #해당 부분 수정 중 0614_1313

    for output_data in output_list:
        key = output_data.get("key")
        if key not in (0, 1):
            print(f"지원되지 않는 key: {key} (0:left, 1:right 만 지원)")
            continue  # 잘못된 키는 무시하고 다음 이미지로

        # 키에 따라 회전 각도 결정
        angle = angle_value if key == 0 else -angle_value
        img_np = output_data.get('image')

        if img_np is None:
            print(f" 이미지 데이터가 없습니다 (key={key})")
            continue

        # 이미지 회전 및 빈 영역 마스크 생성
        new_rgb, depth, mask = rotator.rotate_frame(img_np, angle)
        out_rgb = ((new_rgb[0].cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()

        # 마스크 후처리
        mask_np = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        # PIL 이미지로 변환 후 리사이즈
        init_img = Image.fromarray(out_rgb).convert("RGB").resize((512, 384))
        mask_img = Image.fromarray(mask_np).convert("L").resize((512, 384))

        # Stable Diffusion으로 빈 영역 인페인팅
        result = rotator.inpaint(init_img, mask_img,
                                prompt=prompt, steps=steps, guidance=guidance)

        # 시각화 → 저장 방식 변경 or 제거 필요 (headless 환경 대응 시)
        show_image(result, title=f"images{key} inpainted")

        # Numpy 형태를 BytesIO로 변환
        result_np = np.array(result)
        img_file = rotator.to_bytesio(result_np, filename=f"{key}.jpg")

        # 결과 리스트에 추가 및 완료 메시지 출력
        file.append((f"images{key}", img_file))
        print(f" 이미지 images{key} 변환 및 파일 추가 완료")

        elapsed = time.time() - start_time  # 경과 시간 계산
        print(f"\n discriminator 처리 시간: {elapsed:.2f}초")      

        return file
