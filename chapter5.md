### 色空間
- OpenCVでは，RGB色空間からHSV色空間で色を扱うために，``cv2.cvtColor``を使用する
  - 使い方: ``変数名 = cv2.cvtColor(変数名, cv2.COLOR_BGR2HSV)``
  - ``cv2.COLOR_BGR2HSV``がBGRからHSVに変換する命令
  - ただし，``plt.imshow()``は**データをRGB形式であると判断して表示するため，表示する際はHSV形式をRGB形式に変換してから**``plt.imshow()``を行う

```python
# HSV形式はimshowで表示できない
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR → HSV
plt.imshow(image)
plt.show()
```

```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('boston.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR → HSV
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB) # HSV → RGB

plt.imshow(image)
plt.show()
```

### RGB色空間
- OpenCVでは，RGBでデータを扱う場合，以下の方法でRGBの各値を取得できる
  - 赤画素値: ``RGB画像変数[:,:,0]``
  - 緑画素値: ``RGB画像変数[:,:,1]``
  - 青画素値: ``RGB画像変数[:,:,2]``


```python
import cv2
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR → RGB

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

print('red')
print(red)

print('green')
print(green)

print('blue')
print(blue)
```

```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR → RGB

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

plt.figure(figsize=(10, 3)) # 横10インチ，縦3インチ

plt.subplot(131) # 縦1分割，横3分割の1番目に描画
red_hist = cv2.calcHist([red], [0], None, [256], [0,256])
plt.title('Red Histogram')
plt.plot(red_hist, color='r')

plt.subplot(132) # 縦1分割，横3分割の2番目に描画
green_hist = cv2.calcHist([green], [0], None, [256], [0,256])
plt.title('Green Histogram')
plt.plot(green_hist, color='g')

plt.subplot(133) # 縦1分割，横3分割の3番目に描画
blue_hist = cv2.calcHist([blue], [0], None, [256], [0,256])
plt.title('Blue Histogram')
plt.plot(blue_hist, color='b')

plt.show()
```


- RGBからグレースケールの式は，``Y = 0.2126R + 0.7152G + 0.0722B``であるため，
  - ``グレースケール画素値変数 = 0.2126 * 赤画素値変数 + 0.7152 * 緑画素値変数 + 0.0722 * 青画素値変数``というプログラムはグレースケースへ変換を意味する

```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR → RGB

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

Y = 0.2126 * red + 0.7152 * green + 0.0722 * blue

plt.imshow(Y)
plt.gray()
plt.show()
```

```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR → RGB

red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]
Y = 0.2126 * red + 0.7152 * green + 0.0722 * blue

after_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(10, 4)) # 横10インチ，縦4インチ

plt.subplot(121) # 縦1分割，横2分割の1番目に描画
plt.title('Y = 0.2126R + 0.7152G + 0.0722B')
plt.gray()
plt.imshow(Y)

plt.subplot(122) # 縦1分割，横2分割の2番目に描画
plt.title('cv2.COLOR_RGB2GRAY')
plt.gray()
plt.imshow(after_image)

plt.show()
```

### BGR色空間
- OpenCVでは，BGRでデータを扱う場合，以下の方法でRGBの各値を取得できる
  - 赤画素値: ``BGR画像変数[:,:,2]``
  - 緑画素値: ``BGR画像変数[:,:,1]``
  - 青画素値: ``BGR画像変数[:,:,0]``

```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')

# BGR形式なので
red = image[:,:,2]
green = image[:,:,1]
blue = image[:,:,0]

plt.figure(figsize=(10, 3)) # 横10インチ，縦3インチ

plt.subplot(131) # 縦1分割，横3分割の1番目に描画
red_hist = cv2.calcHist([red], [0], None, [256], [0,256])
plt.title('Red Histogram')
plt.plot(red_hist, color='r')

plt.subplot(132) # 縦1分割，横3分割の2番目に描画
green_hist = cv2.calcHist([green], [0], None, [256], [0,256])
plt.title('Green Histogram')
plt.plot(green_hist, color='g')

plt.subplot(133) # 縦1分割，横3分割の3番目に描画
blue_hist = cv2.calcHist([blue], [0], None, [256], [0,256])
plt.title('Blue Histogram')
plt.plot(blue_hist, color='b')

plt.show()
```

### HSV色空間
- OpenCVでは，HSVでデータを扱う場合，以下の方法でHSVの各値を取得できる
  - 色彩(H): ``HSV画像変数[:,:,0]``
  - 彩度(S): ``HSV画像変数[:,:,1]``
  - 明度(V): ``HSV画像変数[:,:,2]``
- ただし，opencvで色彩(H)を0〜180で扱うため注意が必要

```python
import cv2
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR → HSV

# HSV形式なので
hue = image[:,:,0]
saturation = image[:,:,1]
value = image[:,:,2]

print('hue')
print(hue)

print('saturation')
print(saturation)

print('value')
print(value)
```

```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('nikka.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR → HSV

# HSV形式なので
hue = image[:,:,0]
saturation = image[:,:,1]
value = image[:,:,2]

plt.figure(figsize=(8, 12)) # 横8インチ，縦12インチ
plt.subplot(321)
plt.title('Hue')
plt.gray()
plt.imshow(hue)

plt.subplot(322)
hue_hist = cv2.calcHist([hue], [0], None, [256], [0,256])
plt.xlim(0, 180)
plt.title('Hue Histogram')
plt.plot(hue_hist)

plt.subplot(323)
plt.title('Saturation')
plt.gray()
plt.imshow(saturation)

plt.subplot(324)
saturation_hist = cv2.calcHist([saturation], [0], None, [256], [0,256])
plt.title('Saturation Histogram')
plt.plot(saturation_hist)

plt.subplot(325)
plt.title('Value')
plt.gray()
plt.imshow(value)

plt.subplot(326)
value_hist = cv2.calcHist([value], [0], None, [256], [0,256])
plt.title('Value Histogram')
plt.plot(value_hist)

plt.show()
```

### HSV色空間
- HSV色空間は色彩が0から360の値になるので，色彩の値を絞りこむことで特定の色を抽出できる
  - 色彩の代表値
    - 0付近: 赤色
    - 60付近: 黄色
    - 120付近: 緑色
    - 180付近: シアン色
    - 240付近: 青色
    - 300付近: マゼンタ色
  - ただし，**opencvでは，色彩は以下のように扱う（値が半分になる）**
    - 0付近: 赤色
    - 30付近: 黄色
    - 60付近: 緑色
    - 90付近: シアン色
    - 120付近: 青色
    - 150付近: マゼンタ色

### HSVを使用したマスク画像の作成
- OpenCVの``cv2.inRange``を使用することで，HSVの要素によるマスク画像を作成できる
  - ``hsv下限の値変数 = np.array([色彩の下限値, 彩度の下限値, 明度の下限値], np.uint8)``
  - ``hsv上限の値変数 = np.array([色彩の上限値, 彩度の上限値, 明度の上限値], np.uint8)``
  - ``マスク画像変数 = cv2.inRange(HSV画像変数, hsv下限の値変数, hsv上限の値変数)``
   
- 以下の画像（color-sample.png）を使用して，特定の色だけを抜き出すプログラムを考える

- サンプルプログラム
```python
# color-sample.pngおよびHの調査
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('color-sample.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR → RGB

plt.figure(figsize=(10, 4)) # 横10インチ，縦4インチ
plt.subplot(121)
plt.title('Image')
plt.imshow(image)

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # BGR → HSV
hue = image[:,:,0]
plt.subplot(122)
hue_hist = cv2.calcHist([hue], [0], None, [256], [0,256])
plt.xlim(0, 180)
plt.title('Hue Histogram')
plt.plot(hue_hist)

plt.show()
```

- hueのヒストグラムで，山があるところが色相がある部分である

```python
# 黄色だけ抜き出すマスクを作成する
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('color-sample.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR → HSV

# 黄色はhueのヒストグラムで30付近なので（数値は手作業で求める）
min = np.array([25, 50, 50], np.uint8)
max = np.array([35, 255, 255], np.uint8)
mask = cv2.inRange(image, min, max)

plt.imshow(mask)
plt.gray() # マスク画像は白黒画像なので
plt.show()
```
