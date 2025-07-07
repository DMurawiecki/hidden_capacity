import re

# Читаем файл
with open('notebooks/text_ppl_vs_compression.ipynb', 'r') as f:
    content = f.read()

# Заменяем все вхождения global_color_map[max_length] на global_color_map[int(max_length)]
content = re.sub(r'global_color_map\[max_length\]', 'global_color_map[int(max_length)]', content)

# Записываем исправленный файл
with open('notebooks/text_ppl_vs_compression.ipynb', 'w') as f:
    f.write(content)

print("Исправлен numpy integer issue в notebook!") 