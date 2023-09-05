# Описание
В [model_utils.py](./src/model_utils.py) лежит код для параметризации Conv2d, а также код для параметризации resnet18.

Параметры MLP подбираются исходя из количества весов аппроксимируемого сверточного слоя. Количество параметров MLP относительно количества параметров исходной свертки задается параметром reduction (при reduction=1 количество параметров равно исходному). 
В реализации также есть параметр size_thresh, ограничивающий максимальный размер hidden_size у MLP (он нужен, чтобы избежать ошибок "CUDA OUT OF MEMORY"). Исходя из этих двух параметров количество скрытых слоев и число нейронов в них определяются автоматически.

# Эксперименты

Все эксперименты проводились на 2080Ti 12 GB.

[Ссылка](https://wandb.ai/petili/ReMLP/overview) на проект в wandb.

Для экспериментов был зафиксирован lr=1e-2 (перебор по возможным lr показал, что он наиболее оптимален) и batch_size=384.

Были обучены модели со значениями reduction от 0.05 до 0.3 (при больших значениях не хватает видеопамяти). 

| Model | reduction | val_accuracy |
|------------|------------|------------|
| Original resnet18  | /  | 0.774  |
| Reparam resnet18  | 0.05  | 0.566  |
| Reparam resnet18  | 0.1  | 0.618  |
| Reparam resnet18  | 0.15  | 0.601  |
| Reparam resnet18  | 0.2  | 0.615  |
| Reparam resnet18  | 0.25  | 0.605   |
| Reparam resnet18  | 0.3  | 0.637  |

При увеличение значения reduction качество увеличивается, и при достижении reduction=1 качество должно сравняться с качеством оригинальной модели. 

