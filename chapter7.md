### 画像フィルタリング
- 順番は以下のように行う
  1. フィルタリングを行うカーネルを作成する
  2. ``フィルタリング後画像変数 = cv2.filter2D(画像変数, -1, カーネル変数)``でフィルタリングを行う

### 平均値フィルタ
- サイズ3×3の平均値フィルタで使用するカーネルは各値が1/9である（9 = 3×3）
- サイズ5×5の平均値フィルタで使用するカーネルは各値が1/25である（25 = 5×5）
- **入力画像に平均値フィルタを適用すると基本的にぼやける**

- 実装例
```python
# 3×3の平均値フィルタに使用するカーネル
kernel_33 = np.ones((3,3), np.float32)/9
# 5×5の平均値フィルタに使用するカーネル
kernel_55 = np.ones((5,5), np.float32)/25
```

```python
# 平均値フィルタを行う
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread('nikka.jpeg')

# BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# カーネルの作成
kernel = np.ones((5,5), np.float32)/25

# 平均値フィルタ
after_image = cv2.filter2D(image, -1, kernel)

plt.imshow(after_image)
plt.show()
```

- サンプル
```python
# カーネルの大きさによる違い
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread('nikka.jpeg')

# BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6)) # 横12インチ，縦6インチ

plt.subplot(131)
plt.title('Original')
plt.imshow(image)

plt.subplot(132)
# カーネルの作成
kernel = np.ones((5,5), np.float32)/25
# 平均値フィルタ
after_image = cv2.filter2D(image, -1, kernel)
plt.title('Kernel: 5 * 5')
plt.imshow(after_image)

plt.subplot(133)
# カーネルの作成
kernel = np.ones((9,9), np.float32)/81
# 平均値フィルタ
after_image = cv2.filter2D(image, -1, kernel)
plt.title('Kernel: 9 * 9')
plt.imshow(after_image)

plt.show()
```

```
