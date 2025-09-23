# 言語処理100本ノック 2025

このプロジェクトは 尾崎直観先生作-[言語処理100本ノック 2025](https://nlp100.github.io/2025/ja/index.html) の解答例です。  
プログラミング言語`Python`を利用することを前提としています。

## 環境構築 - Environment Set Up

このリポジトリでPythonスクリプトにて解答を作成しています。

### 1. `uv`のインストール

パッケージの依存関係を`uv`にて管理します。  
各OSにおける`uv`のインストール方法は下記サイトを参照ください。

[Astral Software Inc. Package Manager - uv](https://docs.astral.sh/uv/)

（なお、Google Colablatoryで解答を作成する場合は[本家HPの実行環境](https://nlp100.github.io/2025/ja/prepare.html#google-colaboratory)を参照ください。）

### 2. リポジトリのクローン/ダウンロード

ssh/zipファイルにてリポジトリをローカル環境にクローン/ダウンロードしてください。

### 3. 仮想環境の作成

`pyproject.toml`, `uv.lock`ファイルに基づき仮想環境を作成します。

```bash
uv sync --extra dev
```

適切に仮想環境が作成されれば、`pyproject.toml`, `uv.lock`と同じディレクトリに`.venv`ファイルが作成されます。

>[!WARNING]
> 1. `uv.lock`ファイルは2025年9月時点の依存関係を記述しています。`transformers`, `peft`といったパッケージはリリースが頻繁に行われているため、APIの更新が行われ、ドキュメントとコードの乖離が生じる可能性があります。
>
> 2. また、`mecab-python3`が`numpy`2系に対応していないことから、本プロジェクトも`numpy`1系を採用しています。

## APIキー・アクセストークンの管理

5章でLLMの有料APIを発行しますが、**APIキーはPythonスクリプトにハードコーディングしないでください。**

本プロジェクトでは以下の方法でAPIキーを管理しました。

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

仮想環境を明に有効化して、スクリプトを実行する場合は、以下のコマンドを利用します。

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

## 参考文献

1. [自然言語処理の基礎 (2023) オーム社](https://www.ohmsha.co.jp/book/9784274229008/)
2. [大規模言語モデル入門 (2023) 技術評論社](https://gihyo.jp/book/2023/978-4-297-13633-8)
3. [大規模言語モデル入門Ⅱ〜生成型LLMの実装と評価 (2024) 技術評論社](https://gihyo.jp/book/2024/978-4-297-14393-0)
