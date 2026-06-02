# локальное хранилище на SQLite 

import sqlite3
import pandas as pd
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(__file__), "data_cache.db")  # файл базы рядом с app.py
CHECK_FILE = os.path.join(os.path.dirname(__file__), ".cache_checked")  # маркер "сегодня уже загружали"
MODULES = ["m1", "m2", "m3", "m4", "m5"]


def _conn():
    return sqlite3.connect(DB_PATH)


def _db_exists() -> bool:
    # проверяю есть ли файл базы и все нужные таблицы в ней
    if not os.path.exists(DB_PATH):
        return False
    try:
        with _conn() as con:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
            return set(MODULES).issubset(set(tables["name"].tolist()))
    except Exception:
        return False


def _checked_today() -> bool:
    # читаю дату из файла-маркера и сравниваю с сегодня
    try:
        with open(CHECK_FILE) as f:
            return date.fromisoformat(f.read().strip()) == date.today()
    except Exception:
        return False


def _mark_checked():
    # записываю сегодняшнюю дату в файл-маркер
    with open(CHECK_FILE, "w") as f:
        f.write(str(date.today()))


def _max_date(module: str):
    # смотрю какая последняя дата есть в таблице модуля
    try:
        with _conn() as con:
            row = pd.read_sql(f"SELECT MAX(date) AS d FROM {module}", con)
            val = row["d"].iloc[0]
            return pd.Timestamp(val) if val else None
    except Exception:
        return None


def _save(data: dict):
    # сохраняю все модули в базу, заменяю старые данные новыми
    with _conn() as con:
        for mod, df in data.items():
            if mod in MODULES and df is not None and len(df) > 0:
                df.to_sql(mod, con, if_exists="replace", index=False)
    print(f"[cache] Данные сохранены в {DB_PATH}")


def _load_all() -> dict:
    # читаю все таблицы из базы и возвращаю как словарь DataFrame
    data = {}
    with _conn() as con:
        for mod in MODULES:
            df = pd.read_sql(f"SELECT * FROM {mod}", con)
            df["date"] = pd.to_datetime(df["date"])
            data[mod] = df
    return data


def _fetch_incremental(new_start: str) -> dict:
    # вызываю каждую функцию явно с нужной датой — иначе дефолт не обновится
    from data_fetcher import (fetch_m1_reserves, fetch_m2_repo,
                               fetch_m3_ofz, fetch_m4_tax_calendar, fetch_m5_treasury)
    print(f"[cache] догружаю только новые данные с {new_start}")
    return {
        "m1": fetch_m1_reserves(start=new_start),
        "m2": fetch_m2_repo(start=new_start),
        "m3": fetch_m3_ofz(start_date=new_start),
        "m4": fetch_m4_tax_calendar(start=new_start),
        "m5": fetch_m5_treasury(start=new_start),
    }


def load_data(force_refresh: bool = False) -> dict:
    # решает откуда брать данные — из базы, инкрементально или полный парсинг

    # если сегодня уже загружали — сразу берём из базы
    if not force_refresh and _db_exists() and _checked_today():
        print("[cache] сегодня уже загружали, беру из базы")
        try:
            data = _load_all()
            for mod, df in data.items():
                print(f"  {mod.upper()}: {len(df)} записей, "
                      f"{df['date'].min().date()} – {df['date'].max().date()}")
            return data
        except Exception as e:
            print(f"[cache] Не удалось прочитать базу: {e}, иду парсить")

    # база есть, но сегодня ещё не обновляли — догружаю только недостающее
    if not force_refresh and _db_exists():
        last_dates = {mod: _max_date(mod) for mod in MODULES}
        if all(v is not None for v in last_dates.values()):
            try:
                old_data = _load_all()
                min_last = min(last_dates.values())
                new_start = (min_last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                new_data = _fetch_incremental(new_start)

                # склеиваю старые и новые данные, убираю дубли по дате
                merged = {}
                for mod in MODULES:
                    combined = pd.concat([old_data[mod], new_data[mod]], ignore_index=True)
                    combined["date"] = pd.to_datetime(combined["date"])
                    combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
                    merged[mod] = combined.reset_index(drop=True)

                _save(merged)
                _mark_checked()
                for mod, df in merged.items():
                    print(f"  {mod.upper()}: {len(df)} записей, "
                          f"{df['date'].min().date()} – {df['date'].max().date()}")
                return merged
            except Exception as e:
                print(f"[cache] Инкрементальное обновление не удалось: {e}, делаю полный парсинг")

    # базы нет или force_refresh — полный парсинг с нуля
    print("[cache] загружаю все данные из источников")
    from data_fetcher import load_all_data
    data = load_all_data()
    try:
        _save(data)
        _mark_checked()
    except Exception as e:
        print(f"[cache] Не удалось сохранить в базу: {e}")
    return data
