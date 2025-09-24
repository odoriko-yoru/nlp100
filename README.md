# 言語処理100本ノック 2025

このプロジェクトは 岡崎直観先生作成の[言語処理100本ノック 2025](https://nlp100.github.io/2025/ja/index.html) の解答例です。
プログラミング言語`Python`を利用することを前提としています。


## 概要

[岡崎直観先生](https://www.chokkan.org/)作「言語処理100本ノック 2025」は、自然言語処理に関わる100の問題がnotebook形式で提供されている問題集です。GitHub上に公開されており、Google ColaboratoryやAWS SageMakerを利用して誰でも取り組むことができます。

問題は以下の10章で構成されており、基礎から最新技術まで段階的に学習できる設計になっています。

1. 準備運動
1. UNIXコマンド
1. 正規表現
1. 言語解析
1. 大規模言語モデル
1. 単語ベクトル
1. 機械学習
1. ニューラルネット
1. 事前学習済み言語モデル（BERT型）
1. 事前学習済み言語モデル（GPT型）

## 環境構築

このリポジトリでPythonスクリプトにて解答を作成しています。

### 1. `uv`のインストール

パッケージの依存関係を`uv`にて管理します。  
各OSにおける`uv`のインストール方法は下記サイトを参照ください。

[Astral Software Inc. Package Manager - uv](https://docs.astral.sh/uv/)

（なお、Google Colaboratoryで解答を作成する場合は[本家HP-実行環境の項](https://nlp100.github.io/2025/ja/prepare.html#google-colaboratory)を参照ください。）

### 2. リポジトリのクローン/ダウンロード

ssh/zipファイルにてリポジトリをローカル環境にクローン/ダウンロードしてください。

### 3. 仮想環境の作成

`pyproject.toml`, `uv.lock`ファイルに基づき仮想環境を作成します。

```bash
uv sync --extra dev
```

適切に仮想環境が作成されれば、`pyproject.toml`, `uv.lock`と同じディレクトリに`.venv`ディレクトリが作成されます。

>[!WARNING]
> 1. `uv.lock`ファイルは2025年9月時点の依存関係を記述しています。`transformers`, `peft`といったパッケージはリリースが頻繁に行われているため、APIの更新が行われ、ドキュメントとコードの乖離が生じる可能性があります。
>
> 2. また、`mecab-python3`が`numpy`2系に対応していないことから、本リポジトリも`numpy`1系を採用しています。

## APIキー・アクセストークンの管理

5章でLLMの有料APIを発行しますが、**APIキーはPythonスクリプトにハードコーディングしないでください。**

本リポジトリでは以下の方法でAPIキーを環境変数として管理しました。

ローカル環境
  - `direnv`と`.envrc`

AWS (GPU利用時)
  - Parameter Store

機密情報の管理については[本家HP](https://github.com/nlp100/2025/blob/main/ja/prepare.md#api%E3%82%AD%E3%83%BC%E3%81%AE%E7%AE%A1%E7%90%86)にも記載がありますので、一読ください。また、AWSなどのクラウド環境を利用する際は、各種クラウドで利用可能な機密情報管理サービスにてAPIキーを管理してください。


## スクリプトの実行

以下のコマンドで作成したスクリプトを実行します。

```console
> uv run <path to python script>
```

仮想環境を明示的に有効化して、スクリプトを実行する場合は、以下のコマンドを利用します。

```console
> . .venv/bin/activate
 (nlp100) > python3 <path to python script>
```

## Gitによるスクリプト管理

作成したスクリプトをGit管理する場合は、`ruff`を介した`pre-commit`を利用できます。プロジェクト開始時に以下のコマンドを実行することで設定が完了します。

```console
> uv run pre-commit install
```

また、`pyproject.toml`の`[tool.ruff]`以降を編集してください。

## GPUの利用

[本家HP-ランタイムの選択](https://nlp100.github.io/2025/ja/prepare.html#id2)に記載のある問題はクラウドGPUサーバ上で実行しました。GPUの利用に慣れる良きチャンスであるため、本リポジトリでも当該問題についてはGPUの利用を推奨します。

## 生成AIによるコード生成・補完

[本家HP-生成AIによるコード生成・補完](https://nlp100.github.io/2025/ja/prepare.html#ai)に記載の通り、本リポジトリでも解答作成時はLLMによるコード補完機能をオフにすることを推奨します。


## 改善提案・バグ報告

バグ報告や改善提案は Issue や Pull Request でお知らせいただけますと大変助かります。

## 参考文献

1. [自然言語処理の基礎 (2023) オーム社](https://www.ohmsha.co.jp/book/9784274229008/)
2. [大規模言語モデル入門 (2023) 技術評論社](https://gihyo.jp/book/2023/978-4-297-13633-8)
3. [大規模言語モデル入門Ⅱ〜生成型LLMの実装と評価 (2024) 技術評論社](https://gihyo.jp/book/2024/978-4-297-14393-0)
