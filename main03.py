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
    在完整拼图中匹配模板拼图，考虑旋转和背景处理，并利用彩色信息。
    """
    # 预处理拼图模板，去除背景
    piece_image = preprocess_puzzle_piece(piece_image_path)

    # 加载完整拼图
    full_image = cv2.imread(full_image_path)

    # 使用 SIFT 提取特征
    sift = cv2.SIFT_create()

    # 将拼图块和完整拼图都转换到 Lab 颜色空间，以更好地利用颜色信息
    piece_lab = cv2.cvtColor(piece_image, cv2.COLOR_BGR2Lab)
    full_lab = cv2.cvtColor(full_image, cv2.COLOR_BGR2Lab)

    # 在 Lab 空间的各个通道上提取特征
    keypoints_full = []
    descriptors_full = []
    keypoints_piece = []
    descriptors_piece = []

    for i in range(3):  # L, a, b 三个通道
        # 对完整拼图的第 i 个通道
        kp_full, des_full = sift.detectAndCompute(full_lab[:, :, i], None)
        if des_full is not None and len(kp_full) > 0:
            keypoints_full.extend(kp_full)
            descriptors_full.append(des_full)

        # 对拼图块的第 i 个通道
        kp_piece, des_piece = sift.detectAndCompute(piece_lab[:, :, i], None)
        if des_piece is not None and len(kp_piece) > 0:
            keypoints_piece.extend(kp_piece)
            descriptors_piece.append(des_piece)

    # 检查是否成功检测到特征点
    if len(descriptors_piece) == 0 or len(descriptors_full) == 0:
        raise ValueError("未能检测到足够的特征点，请检查输入图片！")

    # 将描述子列表合并为数组
    descriptors_piece = np.vstack(descriptors_piece)
    descriptors_full = np.vstack(descriptors_full)

    # 特征匹配，使用 FLANN 匹配器和 kNN 方法
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_piece, descriptors_full, k=2)

    # 应用比值测试过滤匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("无法找到足够的匹配点，请检查输入图片！")

    # 提取匹配点位置
    src_pts = np.float32([keypoints_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_full[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用 RANSAC 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # 检查是否找到有效的单应性矩阵
    if M is None or matchesMask.count(1) < 4:
        raise ValueError("未能计算单应性矩阵，请检查匹配点。")

    # 使用单应性矩阵将拼图块的边界映射到完整拼图中
    h, w = piece_image.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # 计算轴对齐的边界矩形
    x, y, w, h = cv2.boundingRect(transformed_corners)

    # 确保坐标为整数并在图像范围内
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    x_end = min(full_image.shape[1], x + w)
    y_end = min(full_image.shape[0], y + h)

    # 绘制匹配结果（以轴对齐的矩形框标记拼图位置）
    marked_image = full_image.copy()
    cv2.rectangle(marked_image, (x, y), (x_end, y_end), (0, 255, 0), 3)

    # 保存结果
    output_path = "marked_image_with_rectangle.jpg"
    cv2.imwrite(output_path, marked_image)
    print(f"匹配结果已保存到: {output_path}")

    # 如果需要，绘制匹配点，方便调试
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    match_img = cv2.drawMatches(piece_image, keypoints_piece, full_image, keypoints_full, good_matches, None, **draw_params)
    cv2.imwrite("color_feature_matches.jpg", match_img)

    return output_path

# 使用範例
output = match_puzzle_with_rotation("IMG_1263.HEIC_compressed.JPEG", "IMG_1264.HEIC_compressed.JPEG")
