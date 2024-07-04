import openai
from dotenv import load_dotenv
import os

# .envファイルから環境変数を読み込み
load_dotenv()

# OpenAI APIキーを設定
openai.api_key = os.getenv('OPENAI_API_KEY')

# データセット
words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential", "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality", "deliberately", "off"]
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential", "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality", "accidentally", "on"]

# プロンプトのテンプレート
eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

# simple_ape関数の呼び出し
result, demo_fn = simple_ape(
    # 入出力のペア
    dataset=(words, antonyms),
    # プロンプトのテンプレート
    eval_template=eval_template,
)

# 結果を出力
print(result)
