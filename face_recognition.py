import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
from scipy.spatial.distance import cosine

class FaceRecognition:
    def __init__(self, database_path, threshold=0.6, model_path="model/vggface2_model.pt"):
        """
        初始化人脸识别系统
        :param database_path: 人脸特征数据库的路径
        :param threshold: 余弦相似度阈值，小于此值认为是同一个人
        """
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化MTCNN和FaceNet
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        # 初始化FaceNet模型
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # 加载人脸特征数据库
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)
            
        self.threshold = threshold
        
    def process_frame(self, frame):
        """处理单帧图像"""
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用MTCNN检测和对齐人脸
        boxes, _ = self.mtcnn.detect(frame_rgb)
        if boxes is None or len(boxes) == 0:
            return []
            
        results = []

        for box in boxes:
            # 确保边界框坐标为整数
            box = box.astype(int)
            
            try:
                # 确保边界框在图像范围内
                h, w = frame_rgb.shape[:2]
                box[0] = max(0, min(box[0], w-1))
                box[1] = max(0, min(box[1], h-1))
                box[2] = max(0, min(box[2], w-1))
                box[3] = max(0, min(box[3], h-1))
                
                # 检查边界框是否有效
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                    
                # 提取人脸区域
                face_img = frame_rgb[box[1]:box[3], box[0]:box[2]]
                if face_img.size == 0:
                    continue
                    
                # 使用MTCNN提取对齐后的人脸
                face = self.mtcnn(face_img)
                if face is None:
                    continue
                    
                # 确保face是一个有效的张量
                if not isinstance(face, torch.Tensor) or face.nelement() == 0:
                    continue
                    
                # 获取特征向量
                with torch.no_grad():
                    embedding = self.facenet(face.unsqueeze(0).to(self.device))
                    
                # 识别人脸
                user, confidence = self.identify_face(embedding)
                results.append({
                    'box': box,
                    'user': user,
                    'confidence': confidence
                })
            except Exception as e:
                print(f"处理人脸时出错: {str(e)}")
                continue
            
        return results
        
    def identify_face(self, embedding):
        """识别人脸，返回最匹配的用户名和相似度"""
        if embedding is None:
            return "unknown", 1.0
            
        min_dist = float('inf')
        matched_user = None
        
        try:
            for user, db_embedding in self.database.items():
                dist = cosine(embedding.cpu().numpy().flatten(), db_embedding.flatten())
                if dist < min_dist:
                    min_dist = dist
                    matched_user = user
        except Exception as e:
            print(f"人脸匹配时出错: {str(e)}")
            return "unknown", 1.0
                
        # 如果相似度大于阈值，认为是陌生人
        if min_dist > self.threshold:
            return "unknown", min_dist
        return matched_user, min_dist
        
    def draw_results(self, frame, results):
        """在图像上绘制识别结果"""
        if frame is None:
            return frame
            
        try:
            for result in results:
                box = result['box']
                user = result['user']
                conf = result['confidence']
                
                x1, y1, x2, y2 = box
                
                # 确保坐标在图像范围内
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # 设置颜色（绿色为认识的人，红色为陌生人）
                color = (0, 255, 0) if user != "unknown" else (0, 0, 255)
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                label = f"{user} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"绘制结果时出错: {str(e)}")
            
        return frame 