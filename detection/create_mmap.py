import hashlib
import mmap
import os
import re
import struct
import subprocess
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import objc
import Quartz
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
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
        create_empty_mmap(mmap_path, size=required_size)
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

    mmap_size = 8 * 6  # 6つのdouble値を保存するためのサイズ
    if not os.path.exists(mmap_path) or os.path.getsize(mmap_path) < mmap_size:
        create_empty_mmap(mmap_path, size=mmap_size)

    with open(mmap_path, "r+b") as f:
        with mmap.mmap(f.fileno(), length=mmap_size, access=mmap.ACCESS_WRITE) as mm:
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


def create_empty_mmap(mmap_path: str, size: int = 48):
    """
    指定されたパスに空のmmapファイルを作成します。
    サイズはデフォルトで48バイト（6 * 8）です。
    """
    with open(mmap_path, "w+b") as f:
        f.write(b"\x00" * size)  # 48バイトのゼロで埋める


def get_ffmpeg_camera_names() -> List[str]:
    """ffmpeg を使って avfoundation カメラ名のリストを取得"""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    output = result.stderr  # ffmpeg はデバイス一覧を stderr に出力

    camera_names = []
    in_video_section = False
    for line in output.splitlines():
        # ビデオデバイスの開始地点
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if in_video_section:
            match = re.search(r"\[(\d+)\] (.+)", line)
            if match:
                print(line)
                index = int(match.group(1))
                name = match.group(2).strip()
                camera_names.append(name)
    return camera_names


def list_available_cameras() -> List[Tuple[int, str]]:
    """OpenCV で開けるカメラと，ffmpeg から取得した名前を対応付けて返す"""
    camera_names = get_ffmpeg_camera_names()
    available_cameras = []

    for i in range(len(camera_names)):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            name = camera_names[i]
            info = f"{name} ({width}x{height} @ {fps}fps)"
            available_cameras.append((i, info))
            cap.release()

    return available_cameras


def select_camera() -> int:
    """利用可能なカメラの一覧を表示し、ユーザーに選択させる"""
    available_cameras = list_available_cameras()
    if not available_cameras:
        raise RuntimeError("利用可能なカメラが見つかりません")

    print("\n利用可能なカメラ:")
    for idx, info in available_cameras:
        print(f"カメラ {idx}: {info}")

    while True:
        try:
            selection = int(input("\n使用するカメラのインデックスを入力してください: "))
            if selection in [idx for idx, _ in available_cameras]:
                return selection
            print("無効なカメラインデックスです。もう一度入力してください。")
        except ValueError:
            print("数値を入力してください。")


def capture_from_camera(camera_index: int) -> np.ndarray:
    """指定されたカメラから画像を取得"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ {camera_index} を開けませんでした")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("カメラからの画像取得に失敗しました")

    return frame


def draw_detection_results(
    img: np.ndarray, centers: list[tuple[int, int]]
) -> np.ndarray:
    """
    検出結果を画像上に描画する

    Args:
        img: 元画像
        centers: 検出された中心点のリスト [(x1,y1), (x2,y2), (x3,y3)]
    Returns:
        描画済みの画像
    """
    result_img = img.copy()

    # 各中心点に丸と順位を描画
    for i, (x, y) in enumerate(centers, 1):
        # 丸を描画
        cv2.circle(result_img, (x, y), 20, (0, 255, 0), 2)
        # 順位の数字を描画
        cv2.putText(
            result_img,
            str(i),
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return result_img


def process_camera(
    model_path: str,
    mmap_path: str,
    camera_index: int,
    conf_threshold: float = 0.1,
    target_bgr: tuple[int, int, int] = (0, 0, 255),
    capture_interval: float = 1.0,
):
    """
    指定のYOLOモデルを使い、カメラからの映像に対して予測を実施。
    取得した中心点の候補点の中から、target_bgrの色に近い上位３件の中心座標を取得し、mmapに書き込む。

    Args:
        model_path: YOLOモデルのパス
        mmap_path: 出力先のmmapファイルのパス
        camera_index: 使用するカメラのインデックス
        conf_threshold: 検出の信頼度閾値
        target_bgr: 比較したい色を (B, G, R) で指定
        capture_interval: カメラからの画像取得間隔（秒）
    """
    model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError(f"カメラ {camera_index} を開けませんでした")

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("カメラからの画像取得に失敗しました")
                continue

            result = model.predict(img, save=False, conf=conf_threshold)[0]
            centers = get_top3_centers_by_color(img, result, target_bgr)

            # 検出結果を描画
            result_img = draw_detection_results(img, centers)

            # ウィンドウに表示
            cv2.imshow("Detection Results", result_img)

            # キー入力待ち（1ms）
            key = cv2.waitKey(1)
            if key == 27:  # ESCキーで終了
                break

            # 検出結果をmmapに保存
            save_top3_to_mmap(centers, mmap_path)

            # 指定間隔待機
            time.sleep(capture_interval)

    finally:
        cap.release()
        cv2.destroyAllWindows()


# メイン処理
if __name__ == "__main__":
    COLOR_MMAP_PATH = os.getenv("COLOR_MMAP_PATH", "")
    POSITION_MMAP_PATH = os.getenv("POSITION_MMAP_PATH", "")
    CAMERA_CAPTURE_INTERVAL = float(os.getenv("CAMERA_CAPTURE_INTERVAL", "1.0"))

    target_bgr = load_color_mmap(mmap_path=COLOR_MMAP_PATH)
    print(f"検出対象の色: {target_bgr}")

    camera_index = select_camera()
    print(f"カメラ {camera_index} を使用します")

    process_camera(
        model_path="./assets/yolov8n.pt",
        mmap_path=POSITION_MMAP_PATH,
        camera_index=camera_index,
        conf_threshold=0.005,
        target_bgr=target_bgr,
        capture_interval=CAMERA_CAPTURE_INTERVAL,
    )
