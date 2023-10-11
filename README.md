# VC2 Remote Rendering

## レポジトリ構成

リポジトリは下記のように構成されている：

```bash
.
├── app
│   ├── render.py
│   └── server.py
├── assets
│   ├── glsl
│   │   ├── fragment_shader.glsl
│   │   └── vertex_shader.glsl
│   └── hdr
│       └── museum_of_ethnography_1k.hdr
├── colab
│   ├── notebook.py
│   ├── tunnel.py
│   └── websocket_server.ipynb
├── deploy
│   ├── Dockerfile
│   ├── nginx.conf
│   └── README.md
├── docs
│   ├── index.html
│   └── session-manager.js
├── tests
│   ├── data
│   │   ├── reference.jpg
│   │   └── test_env_map.hdr
│   ├── __init__.py
│   ├── conftest.py
│   └── test_render.py
├── Makefile
├── one_by_one_push.py
├── README.md
├── requirements_dev.txt
└── requirements.txt

9 directories, 23 files
```

各ファイルの内容を以下に示す：

- app/render.py

  ModernGL という Python モジュールを使用して OpenGL コンテキストを生成する．生成したコンテキストを用いて，レンダリングの処理を実行する．

- app/server.py

  WebSocket サーバを起動する．クライアントからのリクエストに応じて WebSocket のコネクションを確立し，タスクが生成される．このタスクは，キャンセルのリクエストがくるまで停止しない無限ループとなっており，ループ中にレンダリングとその結果画像の送信が実行される．このループ中のレンダリングにおいて，サンプリングは継続される．WebSocket のコネクション確立後，クライアントから何らかのリクエストがあると，タスクをキャンセルして新しいタスクを生成する．新しいタスクが生成された時点で，サンプリング進捗は０に戻る．

- assets/glsl/fragment_shader.glsl

  OpenGL のフラグメントシェーダーの内容が記述されている．GPU パストレーシングが実装されている．

- assets/glsl/vertex_shader.glsl

  OpenGL のバーテックスシェーダーの内容が記述されている．

- assets/hdr/

  イメージベーストライティングに使用する環境マップを格納する．

- colab/notebook.py

  Google Colaboratory で実行可能なノートブックファイル (.ipynb) を生成するスクリプトが記述されている．

- colab/tunnel.py

  Google Colaboratory で Ngrok のトンネルを使用するためのクラスが記述されている．

- colab/websocket_server.ipynb

  自動生成されたノートブックファイル．

- deploy/Dockerfile

  デモサイトをビルドするための Dockerfile．

- deploy/nginx.conf

  デモサイトをホスティングする Nginx の設定ファイル．

- deploy/README.md

  デプロイ手順書．

- docs/index.html

  WebSocket サーバと通信する WEB ページ．

- docs/session-manager.js

  WebSocket サーバに対してリクエストを送信したり，レスポンスを受信するための処理が記述されている．

- tests/

  pytest に使用するディレクトリ

- Makefile

  Python ファイルのフォーマットおよびリントを実行する際のターゲットが記述されている．

- one_by_one_push.py

  `git push` を1コミット毎に実行するスクリプトが記述されている．

- README.md

  本ファイル．

- requirements.txt

  アプリケーションの実行に必要な python のパッケージが記述されている．

- requirements_dev.txt

  アプリケーションの開発に必要な python のパッケージが記述されている．

## 仮想環境の構築

### Miniconda のインストール

[公式サイト](https://docs.conda.io/en/latest/miniconda.html) から環境に応じたインストーラをダウンロードし，インストールする

### 仮想環境の構築

```bash
conda create --name venv python=3.10.12
```

venv は仮想環境の名前で，任意の文字列に変更可能

python version は Google Colaboratory の [Release Notes](https://colab.research.google.com/notebooks/relnotes.ipynb) で書かれているバージョンに合わせる（2023/07/21 時点で 3.10.12）

### 仮想環境の削除

```bash
conda remove -n venv --all
```

### 仮想環境の有効化

```bash
conda activate venv
```

### 仮想環境の無効化

```bash
conda deactivate
```

## 依存パッケージのインストール

仮想環境を有効化した状態で以下のコマンドを実行：
```bash
pip install -r requirements.txt
```

なお，`requirements.txt` は以下のコマンドで生成されている：
```bash
pip freeze > requirements.txt
```

開発環境の場合は，`requirements_dev.txt` を用いてインストールする（black, pyflakes, nbformat, clang-format, pytest の依存パッケージが含まれる）：
```bash
pip install -r requirements_dev.txt
```

## 環境マップのダウンロード

Linux (bash)
```bash
(mkdir -p assets/hdr && cd assets/hdr && curl -O https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/museum_of_ethnography_1k.hdr)
```

Windows (powershell)
```powershell
(New-Item -Itemtype Directory -Force assets/hdr | Out-Null && Push-Location assets/hdr &&  curl -O https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/museum_of_ethnography_1k.hdr && Pop-Location)
```

## サーバの起動

仮想環境を有効化し，依存パッケージもインストールした状態で以下のコマンドを実行：
```bash
python app/server.py
```

すると，ウェブソケットサーバが起動し，リッスン状態になる．

## ブラウザからのアクセス

`docs/index.html` をブラウザで開き，start ボタンを押下すると，レンダリングが開始される．

画像領域にイベントリスナーが存在し，以下の挙動に対応している：
- 左ボタンでドラッグするとカメラ回転
- 右ボタンでドラッグするとカメラ移動

deploy/README.md の手順に従って，ローカルホストにデモサイトをデプロイしてもよい．

また，下記のサイトをデモサイトとして使用することもできる：

- https://msrohkwr.github.io/vc2-remote-rendering/
- https://demo-mafd.onrender.com/

## 開発者向け情報

### Google Colaboratory 用の notebook を生成する

仮想環境を有効化し，依存パッケージもインストールした状態で以下のコマンドを実行：
```bash
make notebook
```

すると，`colab/websocket_server.ipynb` が生成される．

Google Colaboratory に生成されたファイルをアップロードすれば実行できる．

### Python ファイルの Format & Lint

Format: `make black`

Lint: `make pyflakes`

### GLSL ファイルの Format

`make clang-format`

### Pytest の実行

`pytest`

### one-by-one push

`make one-by-one-push`

## 関連資料

国際

- Masaru Ohkawara, Hideo Saito, Issei Fujishiro: “Experiencing GPU path tracing in online courses,” Graphics and Visual Computing, Vol. 4, June 2021 (pre-proof: April 22, 2021), 200022, ISSN 2666-6294 [doi: [10.1016/j.gvc.2021.200022](https://doi.org/10.1016/j.gvc.2021.200022)].

- Masaru Ohkawara, Hideo Saito, Issei Fujishiro: “Experiencing GPU path tracing in online courses,” invited by Eurographics 2021 Education Papers, Vienna (hybrid), May 2021.

国内

- 大河原 将，藤代 一成：「遠隔社会におけるグラフィックスパラダイム」，FIT2022，CI-001，2022年9月

- 大河原 将，斎藤 英雄，藤代 一成：「 GPUパストレーシングを体験するクラウドベースのコンピュータグラフィックス実習教材」，情報処理学会研究報告，Vol. 2020–CG–178，No. 1，慶應義塾大学 (オンライン開催) ，2020年6月
