import pickle
from pathlib import Path
from tqdm import tqdm
import os

# Правильные параметры
model_name = 'EleutherAI/pythia-160m'
max_lengths = [8, 16, 32, 64, 80, 96, 128, 192, 256, 640]  # Те длины, которые у вас есть
N_mem_tokens = 1

data = []

print(f"Текущая директория: {os.getcwd()}")
print(f"Ищем файлы для модели: {model_name}")
print(f"Длины: {max_lengths}")

for max_length in max_lengths:
    # Попробуем разные варианты путей
    paths_to_try = [
        f'./runs/no_density/{model_name}/mem_{N_mem_tokens}_len_{max_length}.pkl',
        f'runs/no_density/{model_name}/mem_{N_mem_tokens}_len_{max_length}.pkl',
        f'./runs/no_density/{model_name}/mem_{N_mem_tokens}_len_{max_length}.pkl',
        os.path.join('runs', 'no_density', model_name, f'mem_{N_mem_tokens}_len_{max_length}.pkl')
    ]
    
    found = False
    for path_str in paths_to_try:
        load_path = Path(path_str)
        print(f"Проверяем: {load_path}")
        
        if load_path.exists():
            print(f"  ✅ Файл найден по пути: {load_path}")
            try:
                d = pickle.load(open(load_path, 'rb'))
                print(f"  ✅ Загружено {len(d)} экспериментов")
                data.extend(d)
                found = True
                break
            except Exception as e:
                print(f"  ❌ Ошибка загрузки: {e}")
        else:
            print(f"  ❌ Файл не найден по пути: {load_path}")
    
    if not found:
        print(f"  ❌ Файл не найден ни по одному из путей")

print(f"\nВсего загружено {len(data)} экспериментов")

if len(data) > 0:
    print(f"Пример данных:")
    print(f"  max_length: {data[0]['max_length']}")
    print(f"  original_loss: {data[0]['original_loss']}")
    print(f"  best_loss: {data[0]['best_loss']}")
    print(f"  best_accuracy: {data[0]['best_accuracy']}") 