from datetime import datetime
from math import pi
from tqdm import tqdm

import gzip
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostError
import plotly.express as px

pd.options.mode.copy_on_write = True


class FounderAnomaly:
    """Класс для поиска аномалий во временных рядах."""

    # Ключ - индекс колонки в данных, значение - название колонки
    NUM_COLS_TO_NAME = {1: "name", 2: "date", 3: "call_count", 4: "total_call_time"}

    # Странные метрики, поэтому их будем дропать сразу
    DROP_METRICS = {"Instrument/gmonit.apm.methods.span-event-data/-accumulate",
                    "Supportability/Java/Collector/Output/Bytes",
                    "Threads/SummaryState/RUNNABLE/Count",
                    "Threads/Time/CPU/Thread-# (#)/SystemTime",
                    "Threads/Time/CPU/Thread-# (#)/UserTime"}

    # Дата для начала создания eval датасета
    DATA_START_EVAL = datetime(2024, 5, 9)

    FORMAT_DATE = "%Y-%m-%d %H:%M:%S"

    MIN_COUNT_ROW = 43000
    HOUR_TO_MINUTE = 60
    DAY_TO_HOUR = 24

    MAX_PERCENTILE = 90
    MIN_PERCENTILE = 10

    MAX_ANOMALY = 450
    MIN_ERROR_FOR_ANOMALY = 45
    MIN_COUNT_METRIC = 5

    TARGET = "call_count"
    FEATURES = ["sin", "cos"]

    DEFAULT_START_DATE = datetime(2024, 4, 1)
    DEFAULT_FINISH_DATE = datetime(2024, 6, 1)

    def __init__(self, path: str):
        """Инициализация класса.

        Args:
            path (str): Путь до файла с данными.
        """
        self.path = path

        df_real = self.get_data()
        train = self.get_train_data(df=df_real)

        # Данные с найденными аномалиями
        self.final_df = self.get_df_predict_anomaly(df_real=df_real, train=train)

    def get_data(self) -> pd.DataFrame:
        """Выкачиваем данные, дропаем ненужные столбы и дубликаты, удаляем метрики с 0 стандартным отклонением.

        Returns:
            pd.DataFrame: Обработанные данные.
        """
        try:
            df = pd.read_csv(gzip.open(self.path, "r"), sep="\t", header=None)
        except FileNotFoundError:
            print("Неверный путь до файла")
            return pd.DataFrame()

        # Оставляем только нужные столбцы
        df = df[[1, 2, 3, 4]].rename(self.NUM_COLS_TO_NAME, axis=1)
        df["date"] = pd.to_datetime(df["date"], format=self.FORMAT_DATE)

        # Убираем метрики, у которых константные значения запросов и времени их обработки
        std_group_name = df.groupby("name")[["call_count", "total_call_time"]].std().reset_index()
        std_group_name = std_group_name.query("call_count != 0 and total_call_time != 0")

        # Удаляем константные метрики
        df = df.merge(std_group_name["name"], how="inner", on="name")

        necessary_metrics = set(df["name"].unique()) - self.DROP_METRICS
        df = df[df["name"].isin(necessary_metrics)]
        df = df.drop_duplicates()

        # Оставим метрики, у которых есть минимальное количество строк
        counts = df.groupby("name")["call_count"].count().reset_index()
        counts = counts.rename({"call_count": "count_row"}, axis=1)

        important_name = counts.query(f"count_row > {self.MIN_COUNT_ROW}")["name"]
        df = df.merge(important_name, how="inner", on="name")

        print(f"Количество строк: {len(df)}")
        return df.drop("total_call_time", axis=1)

    def get_train_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Чистим временной ряд по нижним и верхним персентилям в каждом сете, сгриппированным по часам и минутам.

        Args:
            df (pd.Dataframe): Данные  с выбросами.

        Returns:
            pd.Dataframe: Данные обрезанные по персентилям.
        """
        df = self.add_columns(df)

        print("Подготовка данных для обучения.")

        dfs = []
        times = df.groupby(["hour", "minute"]).count().index

        for name in tqdm(df["name"].unique()):
            df_name = df[df["name"] == name]

            for hour, minute in times:
                df_date = df_name[(df_name["hour"] == hour) & (df_name["minute"] == minute)]

                if not len(df_date):
                    continue

                max_call_count = np.percentile(df_date["call_count"], self.MAX_PERCENTILE)
                min_call_count = np.percentile(df_date["call_count"], self.MIN_PERCENTILE)
                df_date.loc[:, "call_count"] = df_date["call_count"].clip(min_call_count, max_call_count)

                dfs.append(df_date.copy())

        df = pd.concat(dfs)
        df = df.sort_values(["name", "date"], ignore_index=True)
        return df

    def add_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляем колонки необходимые для прогнозирования временных рядов.

        Args:
            df (pd.DataFrame): Данные, в которые надо добавить новые колонки.

        Returns:
            pd.DataFrame: Данные с новыми колонками.
        """
        df["hour"] = df["date"].map(lambda date: date.hour)
        df["minute"] = df["date"].map(lambda date: date.minute)
        df["minute_day"] = df["date"].map(lambda date: date.hour * self.HOUR_TO_MINUTE + date.minute)

        args = df["minute_day"] / (self.DAY_TO_HOUR * self.HOUR_TO_MINUTE) * (2 * pi)
        df["sin"] = np.sin(args)
        df["cos"] = np.cos(args)

        return df

    def get_df_predict_anomaly(self, df_real: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
        """Обучаем модель на необходимых метриках.

        Args:
            df_real (pd.DataFrame): Данные для подсчета ошибки.
            train (pd.DataFrame): Данные для обучения и эвала. Значения метрик в этих данных с клипами.

        Returns:
            pd.DataFrame: ключ - название метрик, значение - обученная модель.
        """
        # Список для датафреймов с предсказаниями и ошибками
        dfs = []

        print("Обучение моделей для каждой метрики.")

        names = train["name"].unique()
        for name in tqdm(names):
            df_name = train[train["name"] == name]

            eval_name = df_name[df_name["date"] > self.DATA_START_EVAL]
            train_name = df_name[df_name["date"] <= self.DATA_START_EVAL]

            # Если все обучающие значения являются константой, пропускаем метрику
            if train_name[self.TARGET].std() == 0:
                continue

            # Инициализация модели и ее обучение
            model = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=6)

            try:
                model.fit(X=train_name[self.FEATURES],
                          y=train_name[self.TARGET],
                          eval_set=(eval_name[self.FEATURES], eval_name[self.TARGET]),
                          verbose=False)
            except CatBoostError:
                print(f"Problem fit with {name}")
                continue

            # Заполняем колонки реальным значением, предиктом, ошибкой и проставляем флаг об аномалии
            df_name.loc[:, self.TARGET] = df_real[df_real["name"] == name][self.TARGET]
            df_name.loc[:, "predict"] = model.predict(df_name[["sin", "cos"]])
            df_name.loc[:, "error"] = df_name.eval(f"({self.TARGET} - predict) / predict * 100")
            df_name.loc[:, "has_anomaly"] = (np.abs(df_name["error"]) > self.MIN_ERROR_FOR_ANOMALY).astype(int)

            # Не добавляем метрику в общий датафрейм, если нашлм слишком много аномалий
            if df_name["has_anomaly"].sum() > self.MAX_ANOMALY:
                continue

            dfs.append(df_name.copy())

        final_df = pd.concat(dfs, ignore_index=True).sort_values(["name", "date"])

        # Находим метрики, у которых одинаковое количество аномалий. Потенциально - данные метрики идентичны
        name_sum_anomaly = final_df.groupby("name")["has_anomaly"].sum().reset_index()
        name_sum_anomaly = name_sum_anomaly.drop_duplicates("has_anomaly")
        final_df = final_df.merge(name_sum_anomaly["name"], how="right", on="name")

        print(f"Using amount of metrics for predict anomaly: {len(final_df['name'].unique())}")

        return final_df

    def get_anomaly(self, start_date: str = None, finish_date: str = None) -> None:
        """Ищем в датасете с размеченными аномалия необходимое окно и нем ищем аномалии.

        Args:
            start_date (str): Дата начала окна (если указан None - окно с начала данных). Пример 2024-04-25 00:00:00
            finish_date (str): Дата окончания окна (если указан None - окно до конца данных). Пример 2024-05-05 12:10:30
        """
        try:
            if start_date:
                start_date = datetime.strptime(start_date, self.FORMAT_DATE)
            else:
                start_date = self.DEFAULT_START_DATE

            if finish_date:
                finish_date = datetime.strptime(finish_date, self.FORMAT_DATE)
            else:
                finish_date = self.DEFAULT_FINISH_DATE
        except ValueError:
            print("Неверный формат ввода даты. Изучите пример!")
            return None

        # Выбираем данные по окну
        df = self.final_df[(self.final_df["date"] >= start_date) & (self.final_df["date"] <= finish_date)]
        df = df.drop_duplicates(["date", "name"])
        amount_names = len(self.final_df["name"].unique())

        if df.empty:
            print("Check using date.")
            return None

        # Берем данные только с аномалиями
        anomaly_df = df[df["has_anomaly"] == 1]
        group_anomaly = anomaly_df.groupby("date")["has_anomaly"].count().reset_index()
        group_anomaly = group_anomaly[group_anomaly["has_anomaly"] > self.MIN_COUNT_METRIC]

        # Расчитываем процент метрик, в которых были найдены аномалии
        final_df = group_anomaly.rename({"has_anomaly": "precent_metric_with_anomaly"}, axis=1).reset_index(drop=True)
        final_df["precent_metric_with_anomaly"] = final_df.eval(f"precent_metric_with_anomaly / {amount_names} * 100")

        anomaly_df = anomaly_df.merge(group_anomaly["date"], how="right", on="date")

        if anomaly_df.empty:
            print("No anomalies were found.")
            return None

        # Создадим несколько графиков для итогового лога
        for name in anomaly_df["name"].unique():
            show = df[df["name"] == name].sort_values("date")
            fig = px.line(show, x="date", y="call_count")

            anomaly_name = show[show["has_anomaly"] == 1]
            fig.add_scatter(x=anomaly_name["date"], y=anomaly_name["call_count"], name=f"Anomaly", mode="markers")
            fig.update_layout(title_text=name)
            fig.write_image(f"{name}_{start_date}_{finish_date}.png")

        final_df.to_csv(f"log_anomaly_{start_date}_{finish_date}.csv", index=False)


def main() -> None:
    path = "/Users/dmitrii.fomichev/Desktop/k/metrics_collector.tsv.gz"
    founder_anomaly = FounderAnomaly(path=path)

    # Пример даты 2024-05-11 12:30:00
    start_date = "2024-04-20 00:00:00"
    finish_date = "2024-05-15 00:00:00"
    founder_anomaly.get_anomaly(start_date=start_date, finish_date=finish_date)


if __name__ == '__main__':
    main()
