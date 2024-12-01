import cv2
import numpy as np

def preprocess_puzzle_piece(piece_image_path, output_path="preprocessed_piece.png"):
    """
    分割拼圖，去除背景，返回處理後的拼圖圖片，並保存結果。
    """
    # 讀取拼圖圖片
    piece_image = cv2.imread(piece_image_path, cv2.IMREAD_UNCHANGED)
    piece_image_origin = piece_image.copy()

    if piece_image is None:
        print("無法讀取圖片，請檢查路徑。")
        return None

    # 如果圖片沒有Alpha通道，添加一個
    if piece_image.shape[2] == 3:
        piece_image = cv2.cvtColor(piece_image, cv2.COLOR_BGR2BGRA)
    
    # 將圖像轉換為灰度
    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)
    
    # 應用高斯模糊以減少噪點
    blurred = cv2.GaussianBlur(piece_gray, (5, 5), 0)
    
    # 使用Otsu's方法進行閾值處理
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 進行形態學操作，填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 找到輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 創建遮罩
        mask = np.zeros_like(piece_gray)
        cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
        
        # 將遮罩應用到Alpha通道
        b_channel, g_channel, r_channel, alpha_channel = cv2.split(piece_image)
        # 如果沒有Alpha通道，創建一個
        if alpha_channel is None or alpha_channel.size == 0:
            alpha_channel = np.ones_like(b_channel) * 255
        alpha_channel[mask == 0] = 0  # 背景部分設置為透明
        piece_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        
        # 裁剪拼圖至其邊界
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = piece_image_origin[y:y+h, x:x+w]
        
        # 保存具有透明背景的裁剪後拼圖
        cv2.imwrite(output_path, cropped_image)
        print(f"預處理後的拼圖已保存到: {output_path}")
        return cropped_image
    else:
        print("無法找到拼圖輪廓")
        return None
    
def match_puzzle_with_rotation(full_image_path, piece_image_path):
    """
    在完整拼圖中匹配模板拼圖，考慮旋轉和背景處理。
    """
    # 預處理拼圖模板，並保存結果
    piece_image = preprocess_puzzle_piece(piece_image_path)

    # 加載完整拼圖
    full_image = cv2.imread(full_image_path)
    full_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)

    # 灰度化模板拼圖
    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)

    # 使用 ORB 提取特徵
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(full_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(piece_gray, None)

    # 特徵匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 過濾匹配點
    good_matches = matches[:10]
    if len(good_matches) < 4:
        raise ValueError("無法找到足夠的匹配點，請檢查輸入圖片！")

    # 提取匹配點位置
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 計算單應性矩陣
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 使用單應性矩陣將拼圖邊界映射到完整圖片
    h, w = piece_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # 繪製匹配結果（以矩形框標記拼圖位置）
    marked_image = full_image.copy()
    x_min, y_min = transformed_corners[:, 0, :].min(axis=0)
    x_max, y_max = transformed_corners[:, 0, :].max(axis=0)

    # 繪製矩形框
    cv2.rectangle(marked_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)

    # 儲存結果
    output_path = "marked_image_with_rectangle.jpg"
    cv2.imwrite(output_path, marked_image)
    print(f"匹配結果已保存到: {output_path}")
    return output_path

# 使用範例
output = match_puzzle_with_rotation("IMG_1263.HEIC_compressed.JPEG", "IMG_1264.HEIC_compressed.JPEG")
