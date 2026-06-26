import os

import langchain_community

target_dir = os.path.join(os.path.dirname(langchain_community.__file__), "chat_models")
target_file = os.path.join(target_dir, "vertexai.py")

content = (
    "class ChatVertexAI:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise NotImplementedError('ChatVertexAI stub - Vertex AI not installed')\n"
)

with open(target_file, "w") as f:
    f.write(content)

print(f"Записан файл-затычка: {target_file}")
