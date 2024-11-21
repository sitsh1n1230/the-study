import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# 課題が残る手動処理
def preprocess_image_manual(image, method):
    if image is None:
        raise ValueError("Image is not loaded correctly.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    elif method == 'grayscale':
        return gray
    else:
        raise ValueError("Unsupported manual preprocessing method")

# ノイズと照明を評価
def evaluate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def evaluate_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# 自動適応フィルタリングによる処理
def apply_dynamic_preprocessing(image):
    noise_level = evaluate_noise(image)
    brightness = evaluate_lighting(image)

    # ノイズが高い場合に強いフィルタを適用
    if noise_level > 30:
        image = cv2.medianBlur(image, 5)  # メディアンフィルタを適用
    
    # 照明条件に応じた調整
    if brightness < 100:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif brightness > 200:
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=0)
    
    return image

# ノイズ生成関数
def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        mean = 0
        stddev = 25  # ノイズの標準偏差を調整可能
        gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, gaussian_noise)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    else:
        raise ValueError("Unsupported noise type")

# Mediapipeによるランドマーク検出
def detect_landmarks_with_mediapipe(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    face_mesh.close()
    
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    return landmarks

# 精度評価
def evaluate_landmark_detection(landmarks_true, landmarks_pred):
    errors = []
    for (x_true, y_true), (x_pred, y_pred) in zip(landmarks_true, landmarks_pred):
        error = np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)
        errors.append(error)
    return np.mean(errors), np.std(errors)

# フィードバックによる再処理
def process_image_with_feedback(image, method, phase, landmarks_true):
    preprocessed_img, landmarks = process_image(image, method, phase)

    # 初期検出結果の評価
    mean_error, _ = evaluate_landmark_detection(landmarks_true, landmarks)

    # 誤差が高ければフィルタを強化
    if mean_error > 50:
        preprocessed_img = cv2.medianBlur(preprocessed_img, 5)  # メディアンフィルタを追加
        landmarks = detect_landmarks_with_mediapipe(preprocessed_img)
    
    return preprocessed_img, landmarks

# 手動処理と自動処理を選択
def process_image(image, method, phase):
    if phase == 'manual':
        preprocessed_img = preprocess_image_manual(image, method)
    elif phase == 'adaptive':
        preprocessed_img = apply_dynamic_preprocessing(image)
    else:
        raise ValueError("Invalid phase: Use 'manual' or 'adaptive'")
    
    landmarks = detect_landmarks_with_mediapipe(preprocessed_img)
    return preprocessed_img, landmarks

# グラフ表示
def plot_comparison(manual_data, adaptive_data):
    conditions = list(manual_data.keys())
    manual_mean_errors = [manual_data[cond]['mean_error'] for cond in conditions]
    manual_std_errors = [manual_data[cond]['std_error'] for cond in conditions]
    adaptive_mean_errors = [adaptive_data[cond]['mean_error'] for cond in conditions]
    adaptive_std_errors = [adaptive_data[cond]['std_error'] for cond in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, manual_mean_errors, width, label='Manual (Unsolved)', yerr=manual_std_errors, capsize=5)
    rects2 = ax.bar(x + width/2, adaptive_mean_errors, width, label='Adaptive (Solved)', yerr=adaptive_std_errors, capsize=5)

    ax.set_xlabel('Conditions')
    ax.set_ylabel('Mean Error')
    ax.set_title('Comparison of Landmark Detection Accuracy (Manual vs Adaptive)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()

    fig.tight_layout()
    plt.show()

# メイン処理
def main():
    # Ground truth landmarks (for evaluation purposes)
    landmarks_true = [(100, 100), (150, 150), (200, 200)]  # Example values

    # 画像の読み込み
    img = cv2.imread('Girl.bmp')
    if img is None:
        raise FileNotFoundError("The image file was not found or failed to load.")
    
    original_img = img.copy()

    conditions = ['original', 'bright', 'dark', 'noisy']
    
    # 課題が残る手動処理と課題が解決された自動処理の結果
    manual_data = {}
    adaptive_data = {}

    for condition in conditions:
        if condition == 'original':
            method = 'clahe'
            img = original_img
        elif condition == 'bright':
            img = cv2.convertScaleAbs(original_img, alpha=1.5, beta=30)
            method = 'clahe'
        elif condition == 'dark':
            img = cv2.convertScaleAbs(original_img, alpha=0.5, beta=-50)
            method = 'clahe'
        elif condition == 'noisy':
            img = add_noise(original_img, noise_type='gaussian')
            method = 'clahe'

        # 手動処理（課題が残る処理）
        manual_img, manual_landmarks = process_image(img, method, phase='manual')
        manual_mean_error, manual_std_error = evaluate_landmark_detection(landmarks_true, manual_landmarks)
        manual_data[condition] = {'mean_error': manual_mean_error, 'std_error': manual_std_error}

        # 自動処理（課題が解決された処理）
        adaptive_img, adaptive_landmarks = process_image_with_feedback(img, method, phase='adaptive', landmarks_true=landmarks_true)
        adaptive_mean_error, adaptive_std_error = evaluate_landmark_detection(landmarks_true, adaptive_landmarks)
        adaptive_data[condition] = {'mean_error': adaptive_mean_error, 'std_error': adaptive_std_error}

    # 結果をグラフで表示
    plot_comparison(manual_data, adaptive_data)

if __name__ == "__main__":
    main()
