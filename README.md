# zoltraak-detecter

## 実行方法

1. `uv run ./detection/create_mmap.py`

## 実行時の注意点

1. `create_mmap.py`内にて、入力画像(`INPUT_IMAGE_PATH`)、座標を保存する mmap(`POSITION_MMAP_PATH`)、杖の先端部の色を保存する mmap(`COLOR_MMAP_PATH`)それぞれのパスを指定する変数があります。実行する環境に合わせて、そこを変更してください。
2. mmap に保存される座標の座標系は、python の座標系です。
3. `.env`を root に配置する必要があります。slack から、`.env`の情報を確認してください。

## 何を行っているのか

1. `COLOR_MMAP_PATH`から、杖の先端の色を取得します。もしファイルがなかった場合は、(0, 0, 255)を杖の先端部の色として採用します。
2. `INPUT_IMAGE_PATH`で与えられた画像に対して、yolov8 を用いて、red_ball の bbox の候補を全て取得します。ここでいう候補とは、`conf_threshold`で指定された基準を上回る bbox をさします。
3. bbox の中心座標を取得します。中心座標には、bbox の重心を利用します。
4. `get_top3_centers_by_color()`に、入力画像(`img`)、中心座標の候補リスト(`result`)と先端部の色(`target_bgr`)を入れ、中心座標の候補リストの中から、先端部の色に近い上位 3 件の座標をリスト形式で取得します。
5. `save_top3_to_mmap()`を実行し、取得した上位 3 件の座標リストを`POSITION_MMAP_PATH`に保存します。

処理部分には適宜コメントを GPT に書いてもらいました。
もし上記の説明でわからない部分があれば、そこで補足をしてもらえると助かります。
