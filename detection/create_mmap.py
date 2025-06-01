import os
import mmap
import struct
from ultralytics import YOLO
import cv2
import numpy as np

def load_yolo_model(model_path: str):
	"""YOLOモデルの読み込み"""
	return YOLO(model_path)

def load_image(image_path: str):
	"""画像ファイルを読み込む"""
	img = cv2.imread(image_path)
	if img is None:
		raise FileNotFoundError(f"画像が見つかりません: {image_path}")
	return img

def save_mmap(center_x: float, center_y: float, mmap_path: str):
	"""mmap に center_x, center_y（double形式）を保存する"""

	size = 8 * 2  # 16 バイト

	with open(mmap_path, "r+b") as f:
		with mmap.mmap(f.fileno(), length=size, access=mmap.ACCESS_WRITE) as mm:
			data = struct.pack("dd", center_x, center_y)
			mm.seek(0)
			mm.write(data)

def _read_mmap_center(mmap_path: str):
	"""
	mmap ファイル (16 バイト) から double × 2 を読み取り、
	(center_x, center_y) を返す。
	ファイルサイズが不足している・読み込み失敗なら None を返す。
	"""
	# 必要バイト数 = 8 (double) × 2 = 16 バイト
	expected_size = 8 * 2

	# ファイルが存在しない or サイズ不足 の場合は None
	if not os.path.exists(mmap_path) or os.path.getsize(mmap_path) < expected_size:
		return None

	try:
		with open(mmap_path, "r+b") as f:
			# mmap 長さを 16 バイトに限定して読み込み
			mm = mmap.mmap(f.fileno(), length=expected_size, access=mmap.ACCESS_READ)
			raw = mm.read(expected_size)
			mm.close()
		# struct.unpack で double × 2 をアンパック
		cx, cy = struct.unpack("dd", raw)
		return (float(cx), float(cy))
	except Exception:
		return None

def predict_yolo(model, image_path: str, conf_threshold: float = 0.1):
	"""YOLOモデルで画像に対して予測を実施する"""
	results = model.predict(image_path, save=False, conf=conf_threshold)
	return results[0]

def select_center_from_mmap_or_red(img: np.ndarray, result, mmap_path: str):
	"""
	1) result.boxes から検出済み箱の中心座標を計算
	2) 各中心座標の（画像中のその位置のピクセル）と「赤 (0,0,255)」との差分距離を算出
	3) mmap_path に有効な (center_x, center_y) が入っていれば、
	   その座標に最も近い 중심を返す
	   └── 距離 = Euclid_distance( (center_x, center_y), (検出された中心) )
	4) mmap が空 or 読み込み失敗なら、赤との差分距離が最小の中心を返す
	"""
	boxes = result.boxes
	if len(boxes) == 0:
		return None

	xyxy = boxes.xyxy.cpu().numpy().astype(int)  # shape=(N,4)
	centers = np.column_stack((
		((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(int),
		((xyxy[:, 1] + xyxy[:, 3]) / 2).astype(int),
	))  # shape=(N,2), dtype=int

	red = np.array([0, 0, 255], dtype=float)
	h, w = img.shape[:2]
	red_distances = []
	for (cx, cy) in centers:
		px = np.clip(cx, 0, w - 1)
		py = np.clip(cy, 0, h - 1)
		pixel = img[py, px].astype(float)  # BGR float
		red_distances.append(np.linalg.norm(pixel - red))
	red_distances = np.array(red_distances)  # shape=(N,)
	ref_center = _read_mmap_center(mmap_path) 

	if ref_center is not None:
		ref_x, ref_y = ref_center
		center_array = centers.astype(float) 
		distances_to_ref = np.linalg.norm(center_array - np.array([ref_x, ref_y]), axis=1)
		best_idx = int(np.argmin(distances_to_ref))
		chosen_center = (int(centers[best_idx, 0]), int(centers[best_idx, 1]))
		return chosen_center
	else:
		best_idx = int(np.argmin(red_distances))
		chosen_center = (int(centers[best_idx, 0]), int(centers[best_idx, 1]))
		return chosen_center
	
def process_directory(model_path: str, file_path: str, mmap_path: str, conf_threshold: float = 0.1):
	"""
	指定のYOLOモデルを使い、入力ディレクトリ内の全画像に対して予測を実施。
	各画像に対して、バウンディングボックスの中心座標に円を描画し、出力ディレクトリに保存する。
	"""
	model = load_yolo_model(model_path)
	file_path = os.path.join(file_path)
	result = predict_yolo(model, image_path=file_path, conf_threshold=conf_threshold)
	img = load_image(file_path)
	center_x, center_y = select_center_from_mmap_or_red(img, result, mmap_path)
	print(f"{center_x}, {center_y}")
	save_mmap(center_x, center_y, mmap_path)
	print(f"検出結果画像を保存しました: {mmap_path}")

# 例: メイン処理
if __name__ == "__main__":
	process_directory(
		model_path='./assets/yolov8n.pt',
		file_path='./assets/images/input/image.png',        # 画像の場所
		mmap_path='C:/Users/{ユーザー名}/mmap.txt',          # 出力ディレクトリ
		conf_threshold=0.1
	)