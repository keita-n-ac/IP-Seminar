### 補足: matplotlibでグラフを並べる方法
- ``plt.figure(figsize=(A, B))``を使用することで，描画の大きさを設定できる
  - 横Aインチ，縦Bインチの意味
- ``plt.subplot()``を使用することで，一つの図の中にグラフを追加できる（流れは以下のとおり）
  1. ``plt.subplot()``で画像を描く場所を指定
  2. ``plt.imshow()``で画像を作成
  3. ``plt.subplot()``で画像を描く別の各場所を指定
  4. ``plt.imshow()``で画像を作成
  5. これを繰り返し，最後に``plt.show()``を行う
- ``plt.subplot(ABC)``と書く
  - ``A``, ``B``, ``C``: それぞれ一桁の正の整数
    - 図を縦にA個，横にB個に分割を行う
    - 左上から数えてC番目の領域を指定

- サンプルプログラム
```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')
imageA = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 6)) # 横10インチ，縦6インチ

# 1つ目の描画
plt.subplot(121) # 縦1分割，横2分割の1番目に描画
plt.title('Color Image')
plt.imshow(imageA)

# 2つ目の描画
plt.subplot(122) # 縦1分割，横2分割の2番目に描画
plt.title('Gray Image')
plt.imshow(imageB)
plt.gray()

# 全体を表示
plt.show()
```

- 出力結果
<img src="./subplot.png" width="75%">


### OpenCVによる二値化
- OpenCVをimportする: import cv2
- ``cv2.threshold(画素値変数, しきい値，しきい値判断で変更する値，二値化の方法)``を使用することで，1行で実行できる
- 使用例（しきい値127を超えている画素を255にする）
  - ``変数1, 変数2 = cv2.threshold(画素値変数, 127, 255, cv2.THRESH_BINARY)``
  - この命令を実行すると，2値化を行い，2つのデータを変数で受け取れる
    - 変数1には二値化に使用したしきい値が代入
    - 変数2には二値化後の画像データが代入
- サンプルプログラム
```python
# サンプルプログラム
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # BGR → グレー

plt.figure(figsize=(10, 6)) # 横10インチ，縦6インチ

# 1つ目の描画
plt.subplot(121) # 縦1分割，横2分割の1番目に描画
plt.title('Gray Image')
plt.gray()
plt.imshow(image)

# 二値化を行う
threshold = 127 # しきい値
value, after_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
# 変数1のvalueにしきい値
# 変数2のafter_imageに二値化画像データ

print('しきい値:', value)

# 2つ目の描画
plt.subplot(122) # 縦1分割，横2分割の2番目に描画
plt.title('Binarization Image')
plt.gray()
plt.imshow(after_image)

# 全体を表示
plt.show()
```
- 出力結果
<img src="./binarization.png" width="75%">

### 二値化方法
- ``cv2.threshold()``で使用できる二値化の方法は以下の通り
  - ``cv2.THRESH_BINARY``
    - 入力画像の画素値がしきい値より大きい場合はしきい値判断で変更する値，それ以外の場合は0にする
  - ``cv2.THRESH_BINARY_INV``
    - 入力画像の画素値がしきい値より大きい場合は0，それ以外の場合はしきい値判断で変更する値にする
  - ``cv2.THRESH_TRUNC``
    - 入力画像の画素値がしきい値より大きい場合はしきい値，それ以外の場合はそのままにする
  - ``cv2.THRESH_TOZERO``
    - 入力画像の画素値がしきい値より大きい場合はそのまま，それ以外の場合は0にする
  - ``cv2.THRESH_TOZERO_INV``
    - 入力画像の画素値がしきい値より大きい場合は0，それ以外の場合はそのままにする
  - ``cv2.THRESH_OTSU``
    - 大津の二値化を行う（しきい値は自動的に求まる）

