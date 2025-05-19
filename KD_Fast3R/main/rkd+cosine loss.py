import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    def __init__(self, distance_weight=1.0, angle_weight=2.0, cosine_weight=1.0):
        super(RKDLoss, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.cosine_weight = cosine_weight

    def pdist(self, e, eps=1e-8):
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
        """
        student, teacher: [N, D] 텐서, N: 배치 크기, D: 특징 차원
        각도 관계 손실 계산
        """
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1)) # [1, b, c'] - [b, 1, c'], [b, b, c']
            norm_td = F.normalize(td, p=2, dim=2) # [b, b, c']
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1) # [b*c'*b]

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss
    
    def cosine_loss(self, student, teacher):
        # 각 배치 feature들끼리 cosine similarity 계산 후 loss로 변환
        student_norm = F.normalize(student, p=2, dim=1)
        teacher_norm = F.normalize(teacher, p=2, dim=1)
        cosine_sim = (student_norm * teacher_norm).sum(dim=1)  # [b]
        loss = 1 - cosine_sim.mean()
        return loss

    def process_model_output(self, model_output):
        features = []
        # pointmap: 5차원 텐서 [sample, b, w, h, c]
        # 모델 출력의 각 항목 처리
        for output_dict in model_output:
            pointmap = output_dict['pts3d_in_other_view'] # [b, w, h, c]
            if len(pointmap.shape) == 4:
                    pointmap = pointmap.permute(0, 3, 2, 1) # [b, c, h, w]
                    weighted_features = pointmap # [b, c, h, w]

                    feature = weighted_features.mean(dim=(2, 3)) # [b, c]
                    features.append(feature)

        # 모든 특징을 연결 features.shape = [s, b, c]
        if features:
            combined_features = torch.cat(features, dim=1) # [b, s*c]
            return combined_features
        else:
            raise ValueError("특징 추출에 실패했습니다. 모델 출력 형식을 확인하세요.")

    def forward(self, student_output, teacher_output):
        student_features = self.process_model_output(student_output) # [b, s*c]
        teacher_features = self.process_model_output(teacher_output) # [b, s*c]

        # distance loss # [b, s*c] == [b, c']
        dist_loss = self.RKDDistance(student_features, teacher_features)

        # angle loss
        angle_loss = self.RKDAngle(student_features, teacher_features)

        # cosine loss
        cos_loss = self.cosine_loss(student_features, teacher_features)

        # 가중치 적용한 최종 손실
        loss = (
            self.distance_weight * dist_loss 
            + self.angle_weight * angle_loss
            + self.cosine_weight * cos_loss
        )

        return loss