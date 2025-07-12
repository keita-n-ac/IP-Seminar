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
