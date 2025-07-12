### OpenCVによる物体検出
- 物体検出を行う場合，学習データ（大量の検出したい物体が映っている画像と検出したい物体が映っていない画像）を用意し，検出したい物体の特徴をコンピュータで学習させる
- 学習用データから得られた特徴をまとめたデータ「カスケード分類器」を作成する
- OpenCVには学習済みカスケード分類器（https://github.com/opencv/opencv/tree/master/data/haarcascades ）が用意されているため，本演習ではこれを使用する
  - カスケード分類器を自作することで，任意の物体を検出するプログラムを作成できる

- 学習済み分類器の一例

| ファイル名 | 対象物体 |
| ---- | ---- |
| ``haarcascade_frontalface_default.xml`` | 正面顔 |
| ``haarcascade_fullbody.xml`` | 全身 |
| ``haarcascade_eye.xml`` | 目 |
| ``haarcascade_frontalcatface.xml``| 正面猫顔 |
| ``haarcascade_upperbody.xml`` | 上半身 |

- 本セミナーでは．``haarcascade_frontalface_default.xml``と``haarcascade_frontalcatface.xml``を使用する

- 適用画像（``person-sample.jpg``）
- 適用画像（``people-sample.jpg``）

### 学習済みカスケード分類器の使い方
- 正面顔を分類する``haarcascade_frontalface_default.xml``を使用する
- 学習済みカスケード分類器を使うためには，``cv2.CascadeClassifier()``を使用する
  - 実装例: ``カスケード分類器変数 = cv2.CascadeClassifier('分類器のファイル名')``
- 分類器にデータを入力する場合，``detectMultiScale()``を使用する
  - 実装例: ``分類結果 = カスケード分類器変数.detectMultiScale(データ変数)``

- ここまでのプログラム
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("person-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

# resultに検出結果が格納される
```

### 分類結果の扱い方
- ``print(len(分類結果変数))``とすることで，検知数を取得できる
- ``print(分類結果変数)``とすることで，検知した結果を取得できる
  - 1つ目の結果にアクセスする場合は``print(分類結果変数[0])``とする
  - 2つ目の結果にアクセスする場合は``print(分類結果変数[1])``とする
  - n番目の結果にアクセスする場合は``print(分類結果変数[n-1])``とする
    - **コンピュータでは0番から数えるため注意が必要**

- ここまでのプログラム
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("person-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

# resultに検出結果が格納される

# resultの中身を見る
print(len(result))
print(result)
print(result[0])
```

- n番目の結果にアクセスした時（``print(分類結果変数[n-1])``）に得られた結果である
  - 出力結果``[整数A, 整数B, 整数C, 整数D]``は検知した長方形領域（バウンディングボックス）のパラメータを示す
    - 整数Aが長方形領域の左上のX座標，アクセス方法: ``分類結果変数[n-1][0]``
    - 整数Bが長方形領域の左上のY座標，アクセス方法: ``分類結果変数[n-1][1]``
    - 整数Cが長方形領域の幅，アクセス方法: ``分類結果変数[n-1][2]``
    - 整数Dが長方形領域の高さ，アクセス方法: ``分類結果変数[n-1][3]``
  - 検知した長方形領域を``cv2.rectangle()``を使用して取り囲んで表示する

- ここまでのプログラム
```python

import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("person-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

# resultに検出結果が格納される

# resultの中身を見る
print(len(result))
print(result)
print(result[0])

x = result[0][0] # 検知領域の左上のX座標
y = result[0][1] # 検知領域の左上のY座標
w = result[0][2] # 検知領域の幅
h = result[0][3] # 検知領域の高さ
cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 10)
plt.imshow(image)
plt.show()
```

- まとめると
```python


# まとめると
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("person-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

x = result[0][0] # 検知領域の左上のX座標
y = result[0][1] # 検知領域の左上のY座標
w = result[0][2] # 検知領域の幅
h = result[0][3] # 検知領域の高さ
cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 10)
plt.imshow(image)
plt.show()
```

- 複数人の場合
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("people-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

print(len(result))
```

- 検知数が3であるため，誤検知していることがわかる
  - そこで一つずつ，検知結果を表示する
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("people-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

# 色を変えながら，各検知結果を示す
x0 = result[0][0]
y0 = result[0][1]
w0 = result[0][2]
h0 = result[0][3]
# 赤色
cv2.rectangle(image, (x0, y0, w0, h0), (255, 0, 0), 10)

x1 = result[1][0]
y1 = result[1][1]
w1 = result[1][2]
h1 = result[1][3]
# 緑色
cv2.rectangle(image, (x1, y1, w1, h1), (0, 255, 0), 10)

x2 = result[2][0]
y2 = result[2][1]
w2 = result[2][2]
h2 = result[2][3]
# 青色
cv2.rectangle(image, (x2, y2, w2, h2), (0, 0, 255), 10)

plt.imshow(image)
plt.show()
```

- 3番目の結果（青枠）が誤検知していることがわかるので，その表示を止める
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# データ読み込み
image = cv2.imread("people-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

# 色を変えながら，各検知結果を示す
x0 = result[0][0]
y0 = result[0][1]
w0 = result[0][2]
h0 = result[0][3]
# 赤色
cv2.rectangle(image, (x0, y0, w0, h0), (255, 0, 0), 10)

x1 = result[1][0]
y1 = result[1][1]
w1 = result[1][2]
h1 = result[1][3]
# 緑色
cv2.rectangle(image, (x1, y1, w1, h1), (0, 255, 0), 10)

plt.imshow(image)
plt.show()
```

- サンプルプログラム（猫の検知）
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み（読み込むファイルを変える）
cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# データ読み込み
image = cv2.imread("green_cat.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

print(len(result))
```

- 検知数が多いため，誤検知していることがわかる
  - そこで結果を表示する
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み（読み込むファイルを変える）
cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# データ読み込み
image = cv2.imread("green_cat.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

x0 = result[0][0]
y0 = result[0][1]
w0 = result[0][2]
h0 = result[0][3]
# 赤枠
cv2.rectangle(image, (x0, y0, w0, h0), (255, 0, 0), 10)

x1 = result[1][0]
y1 = result[1][1]
w1 = result[1][2]
h1 = result[1][3]
# 青枠
cv2.rectangle(image, (x1, y1, w1, h1), (0, 0, 255), 10)

plt.imshow(image)
plt.show()
```

- 2番目の結果（青枠）が誤検知していることがわかるので，その表示を止める
```python
import cv2
import matplotlib.pyplot as plt

# カスケード型分類器の読み込み（読み込むファイルを変える）
cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# データ読み込み
image = cv2.imread("green_cat.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 読み込んだデータを分類器に適用
result = cascade.detectMultiScale(image)

x0 = result[0][0]
y0 = result[0][1]
w0 = result[0][2]
h0 = result[0][3]
# 赤枠
cv2.rectangle(image, (x0, y0, w0, h0), (255, 0, 0), 10)

plt.imshow(image)
plt.show()
```

### YuNetによる顔検出
- 最新の顔検出方法であるYuNetを使用する
  - 顔を囲むバウンディングボックスと右目、左目、鼻、右口角、左口角の5箇所を検出できる
  - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet に詳細が記載されております

- YuNetの使い方
  - 学習済みカスケード分類器を使うためには，``cv2.FaceDetectorYN.create()``を使用する
  - 学習済みファイル: ``face_detection_yunet_2023mar.onnx``
    - 実装例: ``検出器変数 = cv2.FaceDetectorYN.create("学習済みファイル", "", (0, 0))``
  - 検出器にデータを入力する場合，``setInputSize()``と``detect()``を使用する
  - 実装例
    ```python
    検出器変数.setInputSize((データ画像の幅, データ画像の高さ))
    結果変数1, 結果変数2 = 検出器変数.detect(データ変数)
    ```
    - **結果変数2に情報が格納されている**

- ここまでのプログラム
```python
import cv2
import matplotlib.pyplot as plt

# YuNetの読み込み
face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))

# データ読み込み
image = cv2.imread("people-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 入力サイズを指定する
height, width, channel = image.shape
face_detector.setInputSize((width, height))

# 顔を検出する
result1, result2 = face_detector.detect(image)
# 変数result2に結果が格納されている
```


### YuNet結果の扱い方
- ``print(len(結果変数2))``とすることで，検知数を取得できる
- ``print(結果変数2)``とすることで，検知した結果を取得できる
  - 1つ目の結果にアクセスする場合: ``print(結果変数2[0])``とする
  - 2つ目の結果にアクセスする場合: ``print(結果変数2[1])``とする
  - n番目の結果にアクセスする場合: ``print(結果変数2[n-1])``とする
- ここまでのプログラム
```python
import cv2
import matplotlib.pyplot as plt

# YuNetの読み込み
face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))

# データ読み込み
image = cv2.imread("people-sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 入力サイズを指定する
height, width, channel = image.shape
face_detector.setInputSize((width, height))

# 顔を検出する
result1, result2 = face_detector.detect(image)

print(len(result2))
print(result2[0])
print(result2[1])
```
- n番目の結果にアクセスした時（``print(結果変数2[n-1])``）に得られた結果である最初の4つの数値は検知した長方形領域（バウンディングボック）のパラメータを示す
  - 1つ目の値が長方形領域の左上のX座標，アクセス方法: ``結果変数2[n-1][0]``
  - 2つ目の値が長方形領域の左上のY座標，アクセス方法: ``結果変数2[n-1][1]``
  - 3つ目の値が長方形領域の幅，アクセス方法: ``結果変数2[n-1][2]``
  - 4つ目の値が長方形領域の高さ，アクセス方法: ``結果変数2[n-1][3]``
  - 検知した長方形領域を``cv2.rectangle()``を使用して取り囲んで表示する
    - **各値が小数なので，すべて整数型に変更する必要がある**
      - ``int(小数値)``で小数値を整数に変更できる

