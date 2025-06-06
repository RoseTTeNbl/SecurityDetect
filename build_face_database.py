import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle

def img_to_encoding(image_path, mtcnn, facenet, device):
    """将图片转换为人脸特征向量"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测和对齐人脸
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None or len(boxes) == 0:
        return None
        
    # 使用第一个检测到的人脸
    box = boxes[0].astype(int)
    face = mtcnn(image_rgb[box[1]:box[3], box[0]:box[2]])
    if face is None:
        return None
        
    # 获取特征向量
    with torch.no_grad():
        embedding = facenet(face.unsqueeze(0).to(device))
    return embedding.cpu().numpy().flatten()

def build_face_database(main_folder_path, mtcnn, facenet, device):
    """构建人脸特征数据库"""
    database = {}
    
    for person_name in os.listdir(main_folder_path):
        person_folder = os.path.join(main_folder_path, person_name)
        
        if os.path.isdir(person_folder):
            # 规范化名称格式：小写并移除下划线
            clean_name = person_name.strip().replace("_", " ").lower()
            embeddings = []
            
            print(f"处理用户 {clean_name} 的图像...")
            
            for img_name in os.listdir(person_folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, img_name)
                    emb = img_to_encoding(img_path, mtcnn, facenet, device)
                    if emb is not None:
                        embeddings.append(emb)
                        
            if embeddings:
                # 计算平均特征向量
                avg_embedding = np.mean(embeddings, axis=0)
                database[clean_name] = avg_embedding
                print(f"已添加用户 {clean_name} 的特征向量")
            else:
                print(f"警告: 未能为用户 {clean_name} 提取任何有效的人脸特征")
                
    return database

def who_is_it(image_path, database, mtcnn, facenet, device, threshold=0.6):
    """
    识别图片中的人脸身份
    
    参数:
        image_path -- 图片路径
        database -- 包含人脸编码和对应人名的数据库
        threshold -- 距离阈值，大于此值视为陌生人
        
    返回:
        min_dist -- 最小距离
        identity -- 识别出的身份
    """
    # 计算目标图片的特征向量
    encoding = img_to_encoding(image_path, mtcnn, facenet, device)
    if encoding is None:
        return None, "unknown"
        
    # 初始化最小距离和身份
    min_dist = float('inf')
    identity = "unknown"
    
    # 遍历数据库中的所有特征向量
    for name, db_enc in database.items():
        # 计算欧氏距离
        dist = np.linalg.norm(encoding - db_enc)
        
        # 更新最小距离和身份
        if dist < min_dist:
            min_dist = dist
            identity = name
            
    # 如果最小距离大于阈值，认为是陌生人
    if min_dist > threshold:
        identity = "unknown"
        
    print(f"识别结果: {identity}, 距离: {min_dist:.4f}")
    return min_dist, identity

class FaceDatabase:
    def __init__(self, model_path="model/vggface2_model.pt"):
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化MTCNN和FaceNet模型
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
    def build_database(self, users_dir):
        """构建人脸特征数据库"""
        return build_face_database(users_dir, self.mtcnn, self.facenet, self.device)
        
    def save_database(self, database, output_path):
        """保存人脸特征数据库"""
        with open(output_path, 'wb') as f:
            pickle.dump(database, f)
        print(f"数据库已保存到: {output_path}")
        
    def identify_person(self, image_path, database, threshold=0.6):
        """识别人脸身份"""
        return who_is_it(image_path, database, self.mtcnn, self.facenet, self.device, threshold)

if __name__ == "__main__":
    # 初始化人脸数据库构建器
    face_db = FaceDatabase()
    
    # 构建数据库
    users_dir = "users"  # 用户图像目录
    database = face_db.build_database(users_dir)
    
    # 保存数据库
    face_db.save_database(database, "face_database.pkl") 