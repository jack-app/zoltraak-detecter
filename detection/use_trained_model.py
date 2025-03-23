import os
from ultralytics import YOLO
import cv2

def load_yolo_model(model_path: str):
    """YOLOモデルの読み込み"""
    return YOLO(model_path)

def load_image(image_path: str):
    """画像ファイルを読み込む"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    return img

def save_image(img, output_path: str):
    """画像を保存する"""
    cv2.imwrite(output_path, img)

def predict_yolo(model, image_path: str, conf_threshold: float = 0.1):
    """YOLOモデルで画像に対して予測を実施する"""
    results = model.predict(image_path, save=False, conf=conf_threshold)
    # 複数画像の場合は、結果リストの先頭要素を使用（画像が1枚の場合）
    return results[0]

def draw_center_points(img, result):
    """YOLOの予測結果から、各バウンディングボックスの中心座標に円を描画する"""
    for box in result.boxes:
        # bboxの座標を取得 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # 中心座標の計算
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # 中心に小さな円を描画（色: 黄色）
        # バウンディングボックスの描画（オプション）
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.circle(img, (int(center_x), int(center_y)), radius=5, color=(0, 255, 255), thickness=-1)
    return img

def process_directory(model_path: str, input_dir: str, output_dir: str, conf_threshold: float = 0.1):
    """
    指定のYOLOモデルを使い、入力ディレクトリ内の全画像に対して予測を実施。
    各画像に対して、バウンディングボックスの中心座標に円を描画し、出力ディレクトリに保存する。
    """
    # YOLOモデルの読み込み（1回のみ）
    model = load_yolo_model(model_path)

    # 出力ディレクトリの作成（存在しない場合）
    os.makedirs(output_dir, exist_ok=True)

    # 画像ファイルのリストを取得（拡張子で判定）
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)
            print(f"処理中: {input_path}")
            # 画像に対する予測
            result = predict_yolo(model, input_path, conf_threshold)
            # 画像の読み込み
            img = load_image(input_path)
            # 予測結果から中心座標に円を描画
            img = draw_center_points(img, result)
            # 出力パスの作成
            output_path = os.path.join(output_dir, filename)
            # 画像の保存
            save_image(img, output_path)
            print(f"検出結果画像を保存しました: {output_path}")

# 例: メイン処理
if __name__ == "__main__":
    process_directory(
        model_path='./assets/best.pt',
        input_dir='./assets/images/input',                # 入力ディレクトリ
        output_dir='./assets/images/output',          # 出力ディレクトリ
        conf_threshold=0.1
    )