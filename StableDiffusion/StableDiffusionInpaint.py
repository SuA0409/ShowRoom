# 환경 설정 및 사전 설치
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/Model/ShowRoom

!pip install kornia
!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
!pip install diffusers transformers accelerate
!pip install kornia opencv-python pillow
!pip install flask pyngrok

import torch
from diffusers import StableDiffusionInpaintPipeline

# MIDAS 다운로드(offline 옵션 사용)
torch.hub.load("intel-isl/MiDaS", "MiDaS_small", offline=False)

# transforms는 offline 옵션 없이!
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


# 지정된 파일에 코드 저장
%%writefile rotate_and_inpaint.py

# 라이브러리 설치
import os
import cv2
import torch
import numpy as np
import kornia as K
import argparse
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from torch import autocast

# GPU 최적화: CuDNN 벤치마킹을 활성화하여 연산 성능을 극대화
torch.backends.cudnn.benchmark = True


class SimpleRotator:
    """
    SimpleRotator 클래스:
    1. MiDaS 모델을 사용해 깊이 맵을 예측
    2. 예측된 깊이 맵으로 이미지 역투영 → 3D 회전 → 2D 재투영하여 회전된 이미지와 빈 영역 마스크 생성
    3. Stable Diffusion Inpainting 파이프라인을 사용해 빈 영역을 자연스럽게 채움
    """

    def __init__(self, device='cuda', max_depth_m=5.0, depth_model='MiDaS_small'):
        # 디바이스 설정 (GPU 사용 가능 시 'cuda', 아니면 'cpu')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_depth_m = max_depth_m  # 깊이 정규화 최대값 (미터 단위)

        # 1) MiDaS 모델 로딩 (오프라인 캐시에서만)
        print("[*] MiDaS_small을 로컬 캐시에서 불러옵니다...")
        self.midas = torch.hub.load(
            "intel-isl/MiDaS",
            depth_model,
            map_location=self.device,
            force_reload=False,
            trust_repo=True,
            offline=True  # 로컬 캐시에 없으면 오류
        ).to(self.device).eval()
        try:
            # PyTorch 2.0 컴파일 최적화
            self.midas = torch.compile(self.midas)
        except Exception:
            pass

        # MiDaS 전처리(transform) 불러오기 (depth_model에 따라 small_transform 또는 dpt_transform 선택)
        trans = torch.hub.load("intel-isl/MiDaS", "transforms")
        if depth_model == 'MiDaS_small' and hasattr(trans, 'small_transform'):
            self.transform = trans.small_transform
        else:
            self.transform = trans.dpt_transform
        print("[✓] MiDaS_small 로드 완료.")

        # 2) Stable Diffusion Inpainting 파이프라인 로딩 (로컬 캐시에서만)
        print("[*] Stable Diffusion Inpainting을 로컬 캐시에서 불러옵니다...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True  # 로컬 캐시에 없으면 오류
        ).to(self.device)
        print("[✓] SD Inpainting 로드 완료.")

        # 메모리 절약: attention slicing 활성화
        self.pipe.enable_attention_slicing()
        try:
            # xFormers가 설치되어 있으면 메모리 효율적 attention 활성화
            self.pipe.enable_xformers_memory_efficient_attention()
        except ModuleNotFoundError:
            print("Warning: xFormers not installed, using default attention")

        # 스케줄러를 Euler Ancestral로 교체
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def estimate_depth(self, img_rgb: np.ndarray) -> torch.Tensor:
        """
        MiDaS를 사용해 이미지의 깊이 맵을 예측
        입력:
          - img_rgb: HxWx3 numpy 배열 (RGB)
        출력:
          - 1x1xHxW 텐서 (깊이, [0, max_depth_m] 범위)
        """

        H, W = img_rgb.shape[:2]

        # MiDaS 모델 전처리(transform) 적용 후 디바이스로 이동
        inp = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            depth = self.midas(inp)

        # MiDaS 출력이 3차원이면 (H,W,C) 형태인 경우 1채널 차원 추가
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        # 원본 해상도(H, W)로 보간(interpolate)
        depth = torch.nn.functional.interpolate(
            depth, size=(H, W), mode='bicubic', align_corners=False
        )

        # [0,1] 정규화 후 [0, max_depth_m] 범위로 스케일링
        dmin, dmax = depth.min(), depth.max()
        depth = (depth - dmin) / (dmax - dmin)
        return depth * self.max_depth_m

    @staticmethod
    def unproject(depth: torch.Tensor, Kmat: torch.Tensor) -> torch.Tensor:
        """
        깊이 맵(depth)을 카메라 좌표계 3D 포인트로 역투영(Unproject)
        입력:
          - depth: 1x1xHxW 텐서
          - Kmat: 3x3 카메라 내부 행렬
        출력:
          - 1xHxWx3 텐서 (각 픽셀당 3D 좌표)
        """
        _, _, H, W = depth.shape

        # 깊이 값
        zs = depth[0, 0]

        # 픽셀 좌표 생성
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        xs, ys = xs.to(depth.device).float(), ys.to(depth.device).float()

        # x/z, y/z -> x, y 계산
        xz = xs * zs
        yz = ys * zs
        xyz = torch.stack([xz, yz, zs], dim=-1)

        # 카메라 내부 행렬의 역행렬로 변환
        Kinv = torch.inverse(Kmat)
        flat = xyz.view(-1, 3) @ Kinv.T
        return flat.view(1, H, W, 3)

    @staticmethod
    def project(xyz: torch.Tensor, Kmat: torch.Tensor) -> torch.Tensor:
        """
        3D 포인트를 이미지 평면으로 투영하여 grid_sample용 정규화 그리드 생성
        입력:
          - xyz: 1xHxWx3 텐서 (3D 포인트)
          - Kmat: 3x3 카메라 내부 행렬
        출력:
          - 1xHxWx2 텐서 (정규화된 grid 'u','v' 좌표, [-1,1] 범위)
        """
        # xyz 텐서의 배치 크기(B), 높이(H), 너비(W)를 가져옴
        B, H, W, _ = xyz.shape

        # (B, H, W, 3) → (B*H*W, 3)로 펼쳐서 3D 좌표 목록으로 변환
        flat = xyz.view(-1, 3)

        # 각 3D 좌표에 Kmat을 곱해 투영 전 좌표 계산 후 다시 (B, H, W, 3) 형태로 복원
        proj = (flat @ Kmat.T).view(B, H, W, 3)

        # 동차 좌표 분할: (x/z, y/z)
        xy = proj[..., :2] / (proj[..., 2:3] + 1e-8)

        # [-1, 1] 범위로 정규화 (grid_sample이 이 범위를 사용)
        grid = torch.stack([
            2 * xy[..., 0] / (W - 1) - 1,
            2 * xy[..., 1] / (H - 1) - 1
        ], dim=-1)
        return grid

    def rotate_frame(self, img_rgb: np.ndarray, angle_deg: float):
        """
        2D 이미지를 깊이를 이용해 3D로 역투영 → Y축 기준으로 회전 → 2D로 재투영 →
        빈 영역을 마스크 생성하여 반환
        입력:
          - img_rgb: HxWx3 numpy 배열 (RGB)
          - angle_deg: 회전 각도 (degrees, 양수=시계, 음수=반시계)
        출력:
          - new_rgb: 1x3xHxW 텐서 (회전된 이미지)
          - depth: 1x1xHxW 텐서 (깊이 맵)
          - mask: 1x1xHxW 텐서 (빈 영역 마스크, 1=빈 영역)
        """
        # 1) 2D 이미지 → 텐서화 & 정규화 ([-1,1] 범위)
        img = torch.from_numpy(img_rgb).permute(2, 0, 1)[None].float().to(self.device)
        img = img / 127.5 - 1

        # 2) 깊이 예측
        depth = self.estimate_depth(img_rgb).to(self.device)
        H, W = img_rgb.shape[:2]

        # 3) 카메라 내부 행렬 생성 (예시: f = 0.8 * min(W,H))
        f = min(W, H) * 0.8
        Kmat = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=self.device)

        # 4) 3D 포인트 역투영
        xyz = self.unproject(depth, Kmat)

        # 5) Y축 기준 회전 행렬 생성 (axis-angle 방식)
        theta = angle_deg / 180 * np.pi
        axis = torch.tensor([0.0, -1.0, 0.0], device=self.device)
        rotmat = K.geometry.conversions.axis_angle_to_rotation_matrix((axis * theta)[None])

        # 6) 3D 포인트 회전 및 재투영용 정규화 grid 생성
        xyz_rot = torch.matmul(xyz, rotmat[0, :3, :3].T)
        grid = self.project(xyz_rot, Kmat)

        # 7) grid_sample로 회전된 이미지 샘플링
        new_rgb = torch.nn.functional.grid_sample(
            img, grid, align_corners=False, mode='bilinear', padding_mode='zeros'
        )

        # 8) 그리드 외부 좌표를 빈 영역으로 마스크
        mask = ((grid[..., 0].abs() > 1) | (grid[..., 1].abs() > 1)).unsqueeze(0).float()
        return new_rgb, depth, mask

    def inpaint(self, init_image: Image.Image, mask_image: Image.Image,
                prompt: str, steps: int = 50, guidance: float = 8.5) -> Image.Image:
        """
        Stable Diffusion Inpainting 수행
        입력:
          - init_image: PIL.Image (회전 후 리사이즈된 RGB 이미지)
          - mask_image: PIL.Image (회전 후 마스크, L 모드)
          - prompt: 텍스트 프롬프트
          - steps: inference step 수
          - guidance: guidance scale
        출력:
          - PIL.Image (인페인팅 결과)
        """
        with autocast(self.device.type):
            result = self.pipe(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                guidance_scale=guidance,
                num_inference_steps=steps
            ).images[0]
        return result


def main():
    # 명령줄 인자 파싱 (argparse 사용)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--angle", type=float, default=30,
                        help="기본 회전 각도 (절대값). "
                             "ST_result.txt에서 '0'은 +angle, '1'은 -angle로 사용됩니다.")
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50,
                        help="Num inference steps for inpainting")
    parser.add_argument("--guidance", type=float, default=8.5,
                        help="Guidance scale for inpainting")
    parser.add_argument("--prompt", type=str,
                        default="Extend only the background wall and floor. Do not add new objects or decorations. "
                                "Match color and lighting. Keep everything minimal.")
    # 경로 수정
    parser.add_argument("--st-path", type=str,
                        default="/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/ST/ST_result.txt",
                        help="ST_result.txt 경로 (각 줄: '<이미지번호> <0또는1>')")
    parser.add_argument("--img-folder", type=str,
                        default="/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Images",
                        help="원본 이미지가 저장된 폴더 경로 ('0.jpg', '1.jpg', '2.jpg' 등이 있음)")
    parser.add_argument("--out-folder", type=str,
                        default="/content/drive/MyDrive/Colab Notebooks/Model/ShowRoom/Input/Images",
                        help="회전+인페인팅 결과를 저장할 폴더(존재하지 않으면 생성됨)")
    args = parser.parse_args()

    # 출력 폴더가 없으면 생성
    os.makedirs(args.out_folder, exist_ok=True)

    # 랜덤 시드 고정 (재현성 확보)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # SimpleRotator 인스턴스 생성 (로컬 캐시 모델 로딩)
    rotator = SimpleRotator(device='cuda', max_depth_m=args.max_depth)

    # ST_result.txt 읽기 (각 줄: "<이미지번호> <0또는1>")
    try:
        with open(args.st_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"'{args.st_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise RuntimeError(f"ST_result.txt 파싱 중 오류 발생: {e}")

    # 각 줄마다 이미지 처리 반복
    for line in lines:
        try:
            img_id_str, direction_str = line.split()
            img_id = int(img_id_str)
            direction = int(direction_str)
        except Exception:
            print(f"잘못된 형식 건너뜀: '{line}' (예상: '<번호> <0또는1>')")
            continue

        # 이미지 파일 경로 구성 (예: "images/0.jpg")
        input_path = os.path.join(args.img_folder, f"{img_id}.jpg")
        if not os.path.isfile(input_path):
            print(f"입력 이미지가 없습니다: {input_path} (건너뜀)")
            continue

        # 방향에 따라 회전 각도 결정: '0' -> +angle, '1' -> -angle
        angle_deg = +args.angle if direction == 0 else -args.angle

        # 이미지 로드 (BGR) → RGB 변환
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            print(f"이미지를 로드할 수 없습니다: {input_path} (건너뜀)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2) 회전 + 깊이 기반 마스크 생성
        new_rgb, depth, mask = rotator.rotate_frame(img_rgb, angle_deg)

        # 3) 회전된 텐서를 numpy 배열(RGB)로 변환
        out_rgb = ((new_rgb[0].cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()

        # 마스크(Mask)도 numpy 배열로 변환하고, 모폴로지 열기 연산으로 잡음 제거
        mask_np = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

        # 4) 인페인팅을 위해 PIL 이미지로 변환 & 512×512 리사이즈
        init_img = Image.fromarray(out_rgb).convert("RGB").resize((512, 512))
        mask_img = Image.fromarray(mask_np).convert("L").resize((512, 512))

        # 5) 인페인팅 수행
        result = rotator.inpaint(
            init_img, mask_img,
            args.prompt, steps=args.steps, guidance=args.guidance
        )

        # 6) 최종 결과 저장
        output_path = os.path.join(args.out_folder, f"1{img_id}.png")
        result.save(output_path)
        print(f"[완료] 이미지 {img_id}.jpg → 회전 {angle_deg}° → 인페인팅 → 저장: {output_path}")


if __name__ == '__main__':
    main()

# 파일 실행
!python rotate_and_inpaint.py