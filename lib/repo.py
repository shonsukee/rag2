# API nameのファイルを作成して，テキスト情報を保存する
import os
import shutil
import pathlib

API_KEY_PROMPT="API name to be modified"

# 入力クエリとLLMからの回答をファイルに保存
def save_doc_to_text(query, response, api_name):
	stored_path = f"../data/history/db/{api_name}/"
	new_path = f"../data/history/{api_name}/"
	num_of_file = len([name for name in os.listdir(new_path) if os.path.isfile(os.path.join(new_path, name))]) + len([name for name in os.listdir(stored_path) if os.path.isfile(os.path.join(stored_path, name))])

	# ファイルに書き込み
	with open(new_path + f"data_{num_of_file}.txt", 'w', encoding='utf-8') as f:
		f.write("------query-----\n")
		f.write(query)
		f.write("\n\n------response-----\n")
		f.write(response)
	f.close()

# 格納済みデータを移動
def mov_saved_doc(source_path, api_name):
	check_dir = pathlib.Path(source_path)
	for file in check_dir.iterdir():
		if file.is_file():
			shutil.move(os.path.join(file), f"../data/history/db/{api_name}")

# 実行テスト
def main():
	prompt = """### Context
## API name to be modified
SwitchBot API

## Code for previous specification
```
BASE_END_POINT = 'https://api.switch-bot.com'
devices = requests.get(
	url=BASE_END_POINT + '/v1.0/devices',
	headers={
		'Authorization': os.environ['SWITCH_BOT_OPEN_TOKEN']
	}
).json()['body']
```

## Question
- First, you read the code and the context corresponding to the previous specification in turn and identify any new information or changes.
- Be sure to make the modifications within the function.
- You then output the modified code with the new information and changes."""

	response = """BASE_END_POINT = 'https://api.switch-bot.com'
devices = requests.get(
	url=BASE_END_POINT + '/V1.1/devices',
	headers={
		'Authorization': os.environ['SWITCH_BOT_OPEN_TOKEN']
	}
).json()[body]
"""

	doc_list = prompt.split('\n## ')
	li = []

	for text in doc_list:
		li.append(text.split('\n'))

	api_list = list(filter(lambda x: x[0]==API_KEY_PROMPT, li))

	api_name = api_list[0][1].split()[0].lower()
	save_doc_to_text(query=prompt, response=response, api_name=api_name)

if __name__ == "__main__":
	main()
