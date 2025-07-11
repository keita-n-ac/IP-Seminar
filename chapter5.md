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
