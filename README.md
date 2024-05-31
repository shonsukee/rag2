# RAGを用いたSwitchBot APIの自動修復

## About

RAG環境を用いて，事前に格納したDBの情報を基に，生成AIが開発過程をサポートすることができます．
今回は SwitchBot API v1.1 のドキュメントを格納するよう設定していますが，お好みで変更することもできます．

## Premise

- OpenAI APIが必要です（有料）
- Pinecone DBをセットアップしておく必要があります
- PineconeのAPI Key等の情報が必要です（無料枠有）

## Package

- openai		== 1.30.1
- streamlit		== 1.31.1
- langchain		== 0.1.9
- llama-index	== 0.10.4

## Usage

### CUIで実行
```
> python3 chat_with_cui.py
```

### GUIで実行
```
> python3 store.py

> streamlit run chat.py
```
