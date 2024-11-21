import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Mediapipeのセットアップ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 従来のヒストグラム均一化とグレースケール変換
def traditional_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return processed_image

# 適応型ヒストグラム均一化（CLAHE）とカラースペース変換
def advanced_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    yuv = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    processed_image = cv2.cvtColor(yuv, cv2.COLOR_BGR2RGB)
    return processed_image

# ガンマ補正
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ランドマーク検出
def detect_landmarks(image):
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            return landmarks
    return None

# 評価指標の計算
def calculate_metrics(true_landmarks, detected_landmarks):
    mse = mean_squared_error(true_landmarks.flatten(), detected_landmarks.flatten())
    return mse

# 照明条件の変動をシミュレートする関数
def adjust_lighting(image, alpha=1.0, beta=0):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

# ノイズを加える関数
def add_noise(image, mean=0, var=10):
    row, col, ch = image.shape
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gaussian.reshape(row, col, ch)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# テスト画像の読み込み
image = cv2.imread('Girl.bmp')

# 元の画像でのランドマーク検出
original_landmarks = detect_landmarks(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 異なる照明条件と画像品質をシミュレート
images = [
    ('Original', image),
    ('Bright', adjust_lighting(image, alpha=1.5, beta=50)),
    ('Dark', adjust_gamma(image, gamma=0.5)),  # ガンマ補正を適用
    ('Noisy', add_noise(image))
]

# 各条件に対してランドマーク検出と評価
mse_values = []
labels = []

for label, img in images:
    # 従来のアプローチでのランドマーク検出
    traditional_image = traditional_preprocess(img)
    landmarks_traditional = detect_landmarks(traditional_image)

    # 新しいアプローチでのランドマーク検出
    advanced_image = advanced_preprocess(img)
    landmarks_advanced = detect_landmarks(advanced_image)

    # デバッグ用: ランドマーク検出結果を確認
    if landmarks_traditional is None:
        print(f"No landmarks detected for {label} using Traditional Approach.")
    if landmarks_advanced is None:
        print(f"No landmarks detected for {label} using Advanced Approach.")

    # ランドマークが検出された場合のみ結果を追加
    if original_landmarks is not None and landmarks_traditional is not None:
        mse_traditional = calculate_metrics(original_landmarks, landmarks_traditional)
        mse_values.append((label, 'Traditional', mse_traditional))

    if original_landmarks is not None and landmarks_advanced is not None:
        mse_advanced = calculate_metrics(original_landmarks, landmarks_advanced)
        mse_values.append((label, 'Advanced', mse_advanced))

# 結果の表示と精度の比較
plt.figure(figsize=(12, 8))

# オリジナル画像
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
if original_landmarks is not None:
    plt.scatter(original_landmarks[:, 0] * image.shape[1], original_landmarks[:, 1] * image.shape[0], c='r')

# その他の条件の画像
for i, (label, img) in enumerate(images[1:], start=2):
    plt.subplot(2, 4, i)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{label} Image')

# 精度の比較グラフ
labels = [f'{label} - {approach}' for label, approach, _ in mse_values]
mse_vals = [mse for _, _, mse in mse_values]

plt.subplot(2, 1, 2)
plt.bar(labels, mse_vals, color=['blue' if approach == 'Traditional' else 'green' for _, approach, _ in mse_values])
plt.xlabel('Approach and Condition')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Landmark Detection Accuracy under Different Conditions')
plt.xticks(rotation=45)
plt.ylim(0, max(mse_vals) * 1.2)  # グラフのy軸範囲を調整
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

