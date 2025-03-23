import os
from inference_sdk import InferenceHTTPClient
import cv2

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Sgv72eNcLHyCraYXsjWb"
)

def load_image(input_path: str):
    """画像ファイルを読み込む"""
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {input_path}")
    return img

def save_image(img, output_path: str):
    """画像を保存する"""
    cv2.imwrite(output_path, img)

def perform_inference(img, model_id: str):
    """画像に対して推論を実施する"""
    result = CLIENT.infer(img, model_id=model_id)
    return result

def draw_predictions(img, predictions):
    """推論結果のbboxを画像上に描画する"""
    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        # 中心座標から左上・右下の座標を算出
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        # バウンディングボックスの描画
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        # ラベル（クラス名と信頼度）の描画
        label = f'{pred["class"]}: {pred["confidence"]:.2f}'
        cv2.putText(img, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def process_image(input_path: str, output_path: str, model_id: str):
    """
    1枚の入力画像を読み込み、推論結果を描画し、出力画像として保存する。

    Args:
        input_path (str): 入力画像のファイルパス
        output_path (str): 出力画像の保存先ファイルパス
        model_id (str): 使用するモデルID
    """
    # 画像の読み込み
    img = load_image(input_path)

    # 推論の実施
    result = perform_inference(img, model_id=model_id)
    predictions = result.get("predictions", [])

    # 推論結果の描画
    img_with_detections = draw_predictions(img, predictions)

    # 結果画像の保存
    save_image(img_with_detections, output_path)
    print(f"検出結果を {output_path} に保存しました。")

def process_directory(input_dir: str, output_dir: str, model_id: str):
    """
    入力ディレクトリ内のすべての画像に対して処理を行い、出力ディレクトリに結果画像を保存する。

    Args:
        input_dir (str): 画像が保存されているディレクトリのパス
        output_dir (str): 処理結果を保存するディレクトリのパス
        model_id (str): 使用するモデルID
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 有効な画像ファイルの拡張子を定義
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # 入力ディレクトリ内の各画像ファイルに対して処理を実施
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"処理中: {input_path}")
            process_image(input_path, output_path, model_id)

# 例: メイン処理
if __name__ == "__main__":
    process_directory(
        input_dir='./assets/images/input',                # 入力ディレクトリ
        output_dir='./assets/images/output',          # 出力ディレクトリ
        model_id="time-pass-for-just-tesing/1"
    )