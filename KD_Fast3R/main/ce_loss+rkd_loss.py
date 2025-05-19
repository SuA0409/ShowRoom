import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    def __init__(self, distance_weight=1.0, angle_weight=2.0, soft_ce_weight=1.0, temperature=2.0):
        super(RKDLoss, self).__init__()
        self.distance_weight = distance_weight    # RKD 거리 손실 가중치
        self.angle_weight = angle_weight          # RKD 각도 손실 가중치 
        self.soft_ce_weight = soft_ce_weight      # teacher 출력에 대한 CE loss 가중치
        self.temperature = temperature            # 소프트 타겟의 온도 파라미터

    def pdist(self, e, eps=1e-8):
        """피처 간 거리 행렬 계산"""
        # e = [b, c']
        e_square = e.pow(2).sum(dim=1) # b
        prod = e @ e.t() # [b, b]
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps) # [b, 1] + [1, b] - 2 * [b, b]
        # [b, b]

        res = res.sqrt() # [b, b]

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0 # 자신은 0
        return res

    def RKDDistance(self, student, teacher):
        """거리 기반 RKD 손실 계산"""
        # Input [b, c']
        with torch.no_grad():
            t_d = self.pdist(teacher)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

    def RKDAngle(self, student, teacher):
        """각도 기반 RKD 손실 계산"""
        # student, teacher: [N, D] 텐서, N: 배치 크기, D: 특징 차원
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1)) # [1, b, c'] - [b, 1, c'], [b, b, c']
            norm_td = F.normalize(td, p=2, dim=2) # [b, b, c']
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1) # [b*c'*b]

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

    def RKD_loss(self, student_features, teacher_features):
        """RKD 손실(거리+각도) 계산"""
        # 거리 손실
        dist_loss = self.RKDDistance(student_features, teacher_features)

        # 각도 손실
        angle_loss = self.RKDAngle(student_features, teacher_features)
        
        # 가중치 적용한 RKD 손실
        rkd_loss = self.distance_weight * dist_loss + self.angle_weight * angle_loss
        
        return rkd_loss

    def process_model_output(self, model_output):
        """모델 출력에서 특징 추출"""
        features = []
        # 모델 출력의 각 항목 처리
        for output_dict in model_output:
            # Global pointmap과 confidence 처리
            pointmap = output_dict['pts3d_in_other_view']  # [b, w, h, c]
            conf = output_dict['conf']  # [b, w, h]

            if len(pointmap.shape) == 4 and len(conf.shape) == 3:
                pointmap = pointmap.permute(0, 3, 2, 1)  # [b, c, h, w]
                conf = conf.permute(0, 2, 1).unsqueeze(1)  # [b, 1, h, w]

                # Confidence로 가중치 부여
                weighted_features = pointmap * conf  # [b, c, h, w]

                # Confidence의 합으로 나누어 정규화 (0으로 나누는 것 방지)
                conf_sum = conf.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                normalized_features = (weighted_features.sum(dim=(2, 3)) / conf_sum.squeeze(3).squeeze(2))  # [b, c]

                features.append(normalized_features)

        # 모든 특징을 연결
        if features:
            combined_features = torch.cat(features, dim=1)  # [b, s*c]
            return combined_features
        else:
            raise ValueError("특징 추출에 실패했습니다. 모델 출력 형식을 확인하세요.")
            
    def get_logits(self, output):
        """모델 출력에서 로짓 추출"""
        # 모델 출력에서 logits 또는 예측값 추출
        if isinstance(output, list) and len(output) > 0:
        # output이 리스트인지 확인하고, 비어있지 않은지 확인
        # 모델 출력이 dict들의 list 형태로 구성되어 있다고 가정.
            return output[0]['pts3d_in_other_viewd']
            # logits 키가 없지만 pred 키가 있다면, 해당 값 반환
        
        raise ValueError("출력에서 logits를 찾을 수 없습니다. 모델 출력 구조를 확인하세요.")

    def soft_ce_loss(self, student_logits, teacher_logits):
        """Soft CE Loss: Teacher 출력과의 비교 (KL Divergence)"""
        # 소프트맥스를 통과시킬 때 온도로 나누어 소프트한 확률 분포를 얻음
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        log_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL Divergence 계산 (온도의 제곱으로 스케일링)
        kd_loss = F.kl_div(log_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        return kd_loss

    def CE_loss(self, student_output, teacher_output):
        """CE 손실(Soft) 계산"""
        try:
            student_logits = self.get_logits(student_output)
            teacher_logits = self.get_logits(teacher_output)
            
            # Soft CE Loss (Teacher의 출력과의 비교)
            soft_ce_loss = self.soft_ce_loss(student_logits, teacher_logits)
            
            # 가중치를 적용한 최종 CE 손실
            ce_loss = self.soft_ce_weight * soft_ce_loss
            
            return ce_loss
            
        except ValueError as e:
            print(f"CE 손실 계산 중 오류 발생: {e}")
            return 0.0  # CE loss를 계산할 수 없는 경우 0 반환

    def forward(self, student_output, teacher_output):
        """최종 손실 계산"""
        # 1. 특징 추출
        student_features = self.process_model_output(student_output)
        teacher_features = self.process_model_output(teacher_output)
        
        # 2. RKD 손실 계산
        rkd_loss = self.RKD_loss(student_features, teacher_features)
        
        # 3. CE 손실 계산
        ce_loss = self.CE_loss(student_output, teacher_output)
        
        # 4. 최종 손실 계산
        loss = rkd_loss + ce_loss
        
        return loss