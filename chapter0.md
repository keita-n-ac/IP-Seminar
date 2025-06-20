### 身近にある画像解析
- スマートフォン（スマホ）には高性能なコンピュータとカメラが搭載
  - スマホで撮影した写真をコンピュータで処理できる
  - アプリで簡単に画像解析・画像処理を利用できる

- 例:
  - Instagram: 撮影した写真にエフェクトを付けることができる
  - Googleレンズ: カメラに映った文字を翻訳することができる 

**→ これらを可能にするための技術: 画像解析（画像処理）**

### 画像解析（画像処理）
- ホットな研究分野
  - 防犯カメラ: カメラで映った人物の顔や歩き方から個人を特定できる
  - 自動車のカメラシステム: 真上から見たかのような映像を生成する
    - 周辺の車，人物，白線などを認識する

### 人間の感覚器官
- 人間には5種類の感覚（五感）がある
  1. 視覚
  2. 聴覚
  3. 嗅覚
  4. 味覚
  5. 触覚
- 一説には，五感による知覚の割合で視覚は80%以上を占めると言われている
- 物体が反射する光を視細胞が感じることで，視細胞から脳に信号を送ることで，物体の色を認識する

### 視細胞
- 視細胞は主に明るいところで働く「錐体」と， 暗いところで働く「桿体（かんたい）」で構成されている
  - 人間が色を感じるのは錐体の働きである
  - 人間が暗闇でもしばらく経つと周囲を少し見えるようになるのは桿体の働きである
    - ただし，暗闇で色を識別できないのは，錐体の働きが低いため
- 錐体は波長感度特性の違いによって3つに分類される
  - 反射する光は「波」であるため，波長（波の周期的な長さ）が 異なる
    - 青色周辺（短波長）の感度が高いS錐体 (Short)
    - 緑色周辺（中波長）の感度が高いM錐体 (Middle)
    - 赤色周辺（長波長）の感度が高いL錐体 (Long)

### 色の認識
- 3つの錐体がそれぞれの波長感度特性に応じた刺激を受けて， 刺激の大きさに応じて，視神経から脳へ電気信号を送る
  - 脳はその電気信号の強さの割合で色を認識する

- 色の認識の流れ
  1. 錐体で光を電気信号に変換
  2. 視神経が脳に電気信号を送る
  3. 脳が電気信号の強度割合で色を認識

### 画像処理とコンピュータビジョン
- 画像処理（Image Processing）
  - 与えられた画像に対して，処理を行い，処理された画像を出力
- コンピュータビジョン（Computer Vision）
  - 画像をもとにして，撮影対象がどのようになっているかを認識（判別）し，対象の状態をデータで出力する
- 1960年代から研究がスタート
  - 当時は人工衛星が撮影した画像の画質改善，文章を撮影した 画像の文字認識など
- 現在最もホットな研究分野の1つ: ディープラーニング（深層学習）が得意
  - ディープラーニング手法は人間の脳のメカニズムを利用している

### 画像処理プログラミングを行うために
- **OpenCV**
  - Intelが開発したコンピュータビジョンに関するプログラム群 （ライブラリ）
  - 現在はWillow Garage社が開発・管理
  - このライブラリを利用することで，誰でも手軽に画像処理を 行うことができる

- Pythonプログラムの例（グラフ用紙に読み込んだ画像を表示するイメージ）
```python
import cv2                                        # opencvを使用する  
import matplotlib.pyplot as plt                   # matlotlib（画像の出力先をグラフにするため）を使用する
img = cv2.imread('画像ファイル名')                  # 画像ファイルを読み込む
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # グラフに画像を出力する  
plt.show()                                        # グラフを表示する  
```
