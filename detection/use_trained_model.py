import os
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

def save_image(img, output_path: str):
	"""画像を保存する"""
	cv2.imwrite(output_path, img)

def predict_yolo(model, image_path: str, conf_threshold: float = 0.1):
	"""YOLOモデルで画像に対して予測を実施する"""
	results = model.predict(image_path, save=False, conf=conf_threshold)
	return results[0]

def draw_center_points_top3(img, result):
	"""
	中心の色が赤に近い上位3つを選んだ後、
	その中をconfidenceが高い順にソートして描画
	"""
	boxes = result.boxes
	if len(boxes) == 0:
			return img

	# 信頼度取得
	confidences = boxes.conf.cpu().numpy().flatten()  # (N,)

	# 座標・中心点計算
	xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
	centers = np.column_stack((
			((xyxy[:,0] + xyxy[:,2]) / 2).astype(int),
			((xyxy[:,1] + xyxy[:,3]) / 2).astype(int),
	))  # (N,2)

	# 赤色との差分距離算出
	red = np.array([0, 0, 255], dtype=float)
	h, w = img.shape[:2]
	distances = np.array([
			np.linalg.norm(img[np.clip(cy,0,h-1), np.clip(cx,0,w-1)].astype(float) - red)
			for cx, cy in centers
	])

	top_n = min(3, len(distances))
	top_idxs = np.argsort(distances)[:top_n]

	order = np.argsort(confidences[top_idxs])[::-1]
	sorted_top_idxs = top_idxs[order]

	# 描画色リスト (1位→緑, 2位→黄, 3位→赤)
	draw_colors = [(0,255,0),(0,255,255),(255,0,0)]

	for rank, idx in enumerate(sorted_top_idxs):
			x1, y1, x2, y2 = xyxy[idx].astype(int)
			cx, cy = centers[idx]
			color = draw_colors[rank]
			cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
			cv2.circle(img, (cx, cy), radius=5, color=color, thickness=-1)

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
			result = predict_yolo(model, input_path, conf_threshold)
			img = load_image(input_path)
			img = draw_center_points_top3(img, result)
			output_path = os.path.join(output_dir, filename)
			# 画像の保存
			save_image(img, output_path)
			print(f"検出結果画像を保存しました: {output_path}")

# 例: メイン処理
if __name__ == "__main__":
	process_directory(
		model_path='./assets/yolov8n.pt',
		input_dir='./assets/images/input',
		output_dir='./assets/images/output',
		conf_threshold=0.005
	)