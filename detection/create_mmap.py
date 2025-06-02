import mmap
import os
import struct

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def load_yolo_model(model_path: str):
    """YOLOモデルの読み込み"""
    return YOLO(model_path)


def load_image(image_path: str):
    """画像ファイルを読み込む"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    return img


def load_color_mmap(mmap_path: str) -> tuple[int, int, int]:
    """
    Unityによって指定された杖先端部分のcolorを取得する。
    もしcolorが保存されたmmapがなかった場合は、(0, 0, 255) を返す。

    Returns:
        (B, G, R) のタプル（各要素は int）
    """
    # デフォルト値（赤）
    default_color = (0, 0, 255)
    required_size = 8 * 3

    # ファイルが存在しない or サイズ不足の場合はデフォルト返却
    if not os.path.exists(mmap_path):
        return default_color
    if os.path.getsize(mmap_path) < required_size:
        return default_color

    try:
        with open(mmap_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), length=required_size, access=mmap.ACCESS_READ)
            raw = mm.read(required_size)
            mm.close()

        b, g, r = struct.unpack("3d", raw)
        return (int(b), int(g), int(r))
    except Exception:
        return default_color


def save_top3_to_mmap(centers: list[tuple[int, int]], mmap_path: str):
    """
    centers: [(x1,y1),(x2,y2),(x3,y3)] の形式を想定
    mmap_path には 48 バイト (6 * 8) を確保しておく必要があります。
    -------------------------------------------------------------
    savable_data = [x1, y1, x2, y2, x3, y3] をすべて double として書き込みます。
    """
    # 要素が 3 件未満なら、(0.0,0.0) で埋める
    padded = centers + [(0, 0)] * (3 - len(centers))
    flat = []
    for x, y in padded[:3]:
        flat.extend([float(x), float(y)])

    data = struct.pack("6d", *flat)

    with open(mmap_path, "r+b") as f:
        with mmap.mmap(f.fileno(), length=8 * 6, access=mmap.ACCESS_WRITE) as mm:
            mm.seek(0)
            mm.write(data)


def predict_yolo(model, image_path: str, conf_threshold: float = 0.1):
    """YOLOモデルで画像に対して予測を実施する"""
    results = model.predict(image_path, save=False, conf=conf_threshold)
    return results[0]


def get_top3_centers_by_color(
    img: np.ndarray, result, target_bgr: tuple[int, int, int]
) -> list[tuple[int, int]]:
    """
    1) result.boxes から検出済み箱の中心座標を計算
    2) 各中心座標における画像ピクセルと target_bgr 色との差分距離を算出
    3) 距離が小さい上位3件の中心点を (x, y) タプルのリストで返す

    Args:
            img: BGR 画像（NumPy 配列）
            result: 検出結果オブジェクト（boxes プロパティを持つ）
            target_bgr: 比較したい色を (B, G, R) で指定
    Returns:
            上位3件（または存在する数だけ）の中心座標 [(x1, y1), (x2, y2), (x3, y3)]
    """
    boxes = result.boxes
    if len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    print(xyxy)
    centers = np.column_stack(
        (
            ((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(int),
            ((xyxy[:, 1] + xyxy[:, 3]) / 2).astype(int),
        )
    )

    target = np.array(target_bgr, dtype=float)

    h, w = img.shape[:2]
    distances = []

    for cx, cy in centers:
        px = np.clip(cx, 0, w - 1)
        py = np.clip(cy, 0, h - 1)
        pixel = img[py, px].astype(float)
        dist = np.linalg.norm(pixel - target)
        distances.append(dist)

    distances = np.array(distances)

    top3_idxs = np.argsort(distances)[:3]

    return [(int(centers[i, 0]), int(centers[i, 1])) for i in top3_idxs]


def process_directory(
    model_path: str,
    file_path: str,
    mmap_path: str,
    conf_threshold: float = 0.1,
    target_bgr: tuple[int, int, int] = (0, 0, 255),
):
    """
    指定のYOLOモデルを使い、入力ディレクトリ内の全画像に対して予測を実施。
    取得した中心点の候補点の中から、target_bgrの色に近い上位３件の中心座標を取得し、mmapに書き込むよう設計。
    """
    model = load_yolo_model(model_path)
    file_path = os.path.join(file_path)
    result = predict_yolo(model, image_path=file_path, conf_threshold=conf_threshold)
    img = load_image(file_path)
    centers = get_top3_centers_by_color(img, result, target_bgr)
    print(f"{centers}")
    save_top3_to_mmap(centers, mmap_path)
    print(f"検出結果画像を保存しました: {mmap_path}")


# 例: メイン処理
if __name__ == "__main__":
    COLOR_MMAP_PATH = os.getenv(
        "COLOR_MMAP_PATH", ""
    )  # 杖の先端部の色を保存するmmapファイルの場所を指定します。
    INPUT_IMAGE_PATH = os.getenv(
        "INPUT_IMAGE_PATH", ""
    )  # 入力する画像のパスを指定します。
    POSITION_MMAP_PATH = os.getenv(
        "POSITION_MMAP_PATH", ""
    )  # 出力先のmmapファイルの場所を指定します。
    target_bgr = load_color_mmap(mmap_path=COLOR_MMAP_PATH)
    print(target_bgr)
    process_directory(
        model_path="./assets/yolov8n.pt",  # 使用するモデルです。
        file_path=INPUT_IMAGE_PATH,
        mmap_path=POSITION_MMAP_PATH,
        conf_threshold=0.005,
        target_bgr=target_bgr,
    )
