import dill
import os
import logging
import pandas as pd
from pydantic import BaseModel, ValidationError
from json import loads

# Укажем путь к файлам проекта:
path = os.environ.get('PROJECT_PATH', '.')
logging.info(f'Predict_path: {path}')

# Структура файла данных
class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


def predict() -> None:

    # Пути к файлам тестовых данных (по всем подкаталогам папки тестовых
    #  данных)
    paths_data_files = []
    for dirpath, _, filenames in os.walk(f'{path}/data/test/'):
        if dirpath.endswith('/'):
            paths_data_files.extend(
                [f'{dirpath}{fname}' for fname in filenames])
        else:
            paths_data_files.extend(
                [f'{dirpath}/{fname}' for fname in filenames])

    # Проход по файлам данных
    data = pd.DataFrame()
    for fname in paths_data_files:
        # Пробуем открыть и считать файл как строку
        try:
            with open(fname, 'r') as f:
                lines = f.read()
        # Не открылся - лог ошибки и переход к следующему файлу
        except ValueError as e:
            logging.info(e)
            logging.info(f'File {fname} will passed by.')
            continue

        # Содержимое файла - JSON правильной структуры?
        try:
            Form.model_validate_json(lines)
        # Если нет, то - лог ошибки и переход к следующему файлу
        except ValidationError as e:
            logging.info(e)
            logging.info(f'File {fname} will passed by.')
            continue

        # Правильный JSON преобразуем в словарь и добавляем в датафрейм
        data = pd.concat([data, pd.DataFrame(loads(lines), index=[0])])

    if data.shape[0] > 0:
        logging.info(f'The dataset of {data.shape[0]} files was collected \
successfully!')
    else:
        logging.info(f'There are no test data files in {path}/data/test/ !')
        return

    # Загружаем модель
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)
    logging.info('Model_loaded!')

    # Добавим в датафрейм колонку предсказаний по модели
    data['pred'] = model['pipeline'].predict(data)

    # Запишем результат (2е колонки датафрейма) в файл
    data[['id', 'pred']].to_csv(
        f'{path}/data/predictions/preds.csv', index=False)

    logging.info(f'The predictions were saved in \
{path}/data/predictions/preds.csv')


if __name__ == '__main__':
    predict()
