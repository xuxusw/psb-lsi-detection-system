import requests
import re
import glob
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import time
import os


# заголовки для запросов, чтобы сайты не блокировали как бота
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Referer": "https://www.cbr.ru/"
}

DATE_START = "2016-01-01"
DATE_END = None

# ссылки на источники данных
M1_RUONIA = "https://www.cbr.ru/hd_base/ruonia/dynamics/"
M1_RESERVES = "https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx"
M2_REPO_URL = "https://www.cbr.ru/hd_base/repo/"
M2_KEYRATE_URL = "https://www.cbr.ru/hd_base/keyrate/"
M5_BLIQUIDITY = "https://www.cbr.ru/hd_base/bliquidity/"
M5_SORS_URL = "https://www.cbr.ru/statistics/bank_sector/sors/"
M5_ROSKAZNA_URL = "https://roskazna.gov.ru/finansovye-operacii/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah"


# генерирую сетку рабочих дней для синтетических данных
def _business_days(start=DATE_START, end=DATE_END):
    if end is None:
        end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"date": idx})


# поднимаю значения вокруг стресс-дат для синтетики
def _inject_stress(series: pd.Series, dates, level=3.0, width=20):
    s = series.copy()
    for d in dates:
        d = pd.Timestamp(d)
        mask = (series.index >= d - pd.Timedelta(days=5)) & \
               (series.index <= d + pd.Timedelta(days=width))
        peak = np.linspace(0, level, mask.sum() // 2 + 1)
        peak = np.concatenate([peak, peak[::-1]])[:mask.sum()]
        s[mask] = s[mask] + peak
    return s


# известные стресс-эпизоды — использую как ground truth
STRESS_DATES = ["2014-12-16", "2022-02-24", "2023-08-14"]


# формирую параметры запроса к сайту ЦБ
def _build_cbr_params(start, end):
    return {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": pd.Timestamp(start).strftime("%d.%m.%Y"),
        "UniDbQuery.To": pd.Timestamp(end).strftime("%d.%m.%Y"),
    }


# ищу нужную колонку по ключевым словам в заголовке
def _find_col(df, *keywords):
    kws = [k.lower() for k in keywords]
    for i, col in enumerate(df.columns):
        levels = [str(c).lower() for c in (col if isinstance(col, tuple) else [col])]
        full = " ".join(levels)
        if all(k in full for k in kws):
            return i
    return None


# М1: загружаю обязательные резервы и RUONIA с сайта ЦБ

def fetch_m1_reserves(start=DATE_START, end=DATE_END) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.today().normalize()

    try:
        r = requests.get(M1_RESERVES, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_excel(BytesIO(r.content), header=2)
        df.columns = ["date", "actual_reserves", "required_reserves",
                      "reserve_account"] + list(df.columns[4:])
        df = df[["date", "actual_reserves", "required_reserves"]].dropna()
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["spread"] = df["actual_reserves"] - df["required_reserves"]

        ruonia = fetch_ruonia(start=start, end=end)

        if ruonia.empty:
            print("М1: RUONIA не загружена, создаю колонку с NaN")
            df["ruonia"] = float("nan")
        else:
            # мержу RUONIA по дате, пропуски заполняю предыдущим значением
            df = df.merge(ruonia[["date", "ruonia"]], on="date", how="left")
            df["ruonia"] = df["ruonia"].ffill()

        print("М1: данные загружены с сайта ЦБ")
        return df.sort_values("date").reset_index(drop=True)

    except Exception as e:
        print(f"М1: ЦБ недоступен ({e}), использую синтетические данные")
        return _synthetic_m1()


# загружаю RUONIA отдельно — нужна и для М1
def fetch_ruonia(start=DATE_START, end=DATE_END):
    try:
        r = requests.get(M1_RUONIA, params=_build_cbr_params(start, end), headers=HEADERS, timeout=20)
        r.raise_for_status()

        tables = pd.read_html(StringIO(r.text), decimal=",", thousands=" ")
        if not tables:
            raise ValueError("таблиц не найдено")
        df = tables[0]

        # ищу колонку со ставкой
        def _find_col(df, *keywords):
            kws = [k.lower() for k in keywords]
            for i, col in enumerate(df.columns):
                levels = [str(c).lower() for c in (col if isinstance(col, tuple) else [col])]
                full = " ".join(levels)
                if all(k in full for k in kws):
                    return i
            return None

        idx_ruonia = _find_col(df, "ruonia")
        if idx_ruonia is None:
            idx_ruonia = 1

        df_out = pd.DataFrame({
            "date": df.iloc[:, 0],
            "ruonia": df.iloc[:, idx_ruonia],
        })

        df_out["ruonia"] = pd.to_numeric(df_out["ruonia"], errors="coerce")
        df_out["date"] = pd.to_datetime(df_out["date"], dayfirst=True, errors="coerce")
        df_out = df_out.dropna(subset=["date", "ruonia"]).sort_values("date")

        print("RUONIA: данные загружены с сайта ЦБ")
        return df_out.reset_index(drop=True)

    except Exception as e:
        print(f"ошибка при загрузке RUONIA: {e}")
        raise


# синтетика М1 — использую если ЦБ недоступен
def _synthetic_m1() -> pd.DataFrame:
    np.random.seed(42)
    bdays = _business_days()
    n = len(bdays)

    actual = 2800 + np.cumsum(np.random.normal(0, 15, n))
    actual = np.clip(actual, 1500, 5500)
    required = actual * np.random.uniform(0.82, 0.88, n)
    spread = actual - required

    # RUONIA исторически в диапазоне 6–21%
    ruonia_base = 7.5 + 3 * np.sin(np.linspace(0, 6 * np.pi, n))
    ruonia = ruonia_base + np.random.normal(0, 0.3, n)
    ruonia = np.clip(ruonia, 5, 25)

    df = bdays.copy()
    df["actual_reserves"] = actual
    df["required_reserves"] = required
    df["spread"] = spread
    df["ruonia"] = ruonia

    df = df.set_index("date")
    df["spread"] = _inject_stress(df["spread"], STRESS_DATES, level=400, width=30)
    df["ruonia"] = _inject_stress(df["ruonia"], STRESS_DATES, level=10, width=25)
    return df.reset_index()


# М2: загружаю ключевую ставку — нужна для расчёта спреда репо

def fetch_key_rate(start=DATE_START, end=DATE_END):
    if end is None:
        end = pd.Timestamp.today().normalize()

    r = requests.get(M2_KEYRATE_URL, params=_build_cbr_params(start, end),
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text), decimal=",", thousands=" ")
    df = tables[0]

    # ищу колонку со ставкой
    idx_rate = _find_col(df, "ключевая", "ставка") or _find_col(df, "ставка")
    if idx_rate is None:
        idx_rate = 1

    df_out = pd.DataFrame({
        "date": df.iloc[:, 0],
        "key_rate": df.iloc[:, idx_rate]
    })
    df_out["date"] = pd.to_datetime(df_out["date"], dayfirst=True, errors="coerce")
    df_out["key_rate"] = pd.to_numeric(df_out["key_rate"], errors="coerce")
    df_out = df_out.dropna(subset=["date", "key_rate"]).sort_values("date")
    return df_out.reset_index(drop=True)


# М2: загружаю аукционы репо ЦБ — два этапа: список дат, потом детали по каждой

def fetch_m2_repo(start=DATE_START, end=DATE_END):
    if end is None:
        end = pd.Timestamp.today().normalize()

    try:
        print("этап 1: получаю список дат с аукционами")
        params_all = _build_cbr_params(start, end)
        params_all["UniDbQuery.P1"] = "5"
        r = requests.get(M2_REPO_URL, params=params_all, headers=HEADERS, timeout=20)
        r.raise_for_status()
        tables = _safe_read_html(r.text)
        if not tables:
            print("нет данных об аукционах за период")
            return pd.DataFrame()

        df_overview = tables[0]
        idx_date = _find_col(df_overview, "дата")
        idx_period = _find_col(df_overview, "срок")
        # беру только 7-дневные аукционы — основной инструмент по ТЗ
        df_overview = df_overview[df_overview.iloc[:, idx_period] == 7]
        if idx_date is None:
            idx_date = 0

        dates_series = pd.to_datetime(df_overview.iloc[:, idx_date], dayfirst=True, errors="coerce")
        unique_dates = sorted(dates_series.dropna().dt.date.unique())
        print(f"найдено {len(unique_dates)} уникальных дат с аукционами")

        if not unique_dates:
            return pd.DataFrame()

        print("этап 2: загружаю детали по каждой дате")
        all_records = []
        for d in unique_dates:
            print(d)
            try:
                params_day = _build_cbr_params(d, d)
                params_day["UniDbQuery.ShowAll"] = "1"
                params_day["UniDbQuery.P1"] = "5"
                r = requests.get(M2_REPO_URL, params=params_day, headers=HEADERS, timeout=10)
                r.raise_for_status()
                tables = _safe_read_html(r.text)
                if tables:
                    df_day = tables[0]
                    if df_day.shape[0] > 1:
                        records = _parse_vertical_day_table(df_day, pd.Timestamp(d))
                        all_records.extend(records)
            except Exception as e:
                print(f"ошибка при обработке {d}: {e}")

        if not all_records:
            return pd.DataFrame()

        df_out = pd.DataFrame(all_records)
        df_out = _finalize_auction_df(df_out, start, end)
        print(f"итого загружено {len(df_out)} записей аукционов")
        return df_out

    except Exception as e:
        print(f"М2: ЦБ недоступен ({e}), использую синтетические данные")
        return _synthetic_m2()


# безопасно читаю HTML-таблицы, возвращаю пустой список если ничего нет
def _safe_read_html(html_text):
    try:
        return pd.read_html(StringIO(html_text), decimal=",", thousands=" ")
    except ValueError:
        return []
    except Exception:
        return []


# парсю вертикальную таблицу одного аукционного дня
def _parse_vertical_day_table(df, date):
    records = []
    current = {}
    col_label = 0
    col_value = 1
    if df.shape[1] < 2:
        return records

    for _, row in df.iterrows():
        label = str(row.iloc[col_label]).strip().lower()
        value = row.iloc[col_value]

        # пропускаю пустые строки и служебные заголовки
        if label in ("", "nan", "параметр", "значение"):
            if current and current.get("demand") is not None:
                records.append(current)
                current = {}
            continue

        # новый аукцион начинается со строки "Тип аукциона"
        if "тип аукциона" in label:
            if current and current.get("demand") is not None:
                records.append(current)
                current = {}
            current = {"auction_type": str(value).strip()}
            continue

        # извлекаю нужные параметры по ключевым словам
        if "объем спроса" in label and "заключенных" not in label:
            current["demand"] = _to_num(value)
        elif ("общий объем заключенных сделок" in label or
              ("объем заключенных сделок" in label and "в рамках лимита" not in label)):
            current["placement"] = _to_num(value)
        elif "ставка отсечения" in label:
            current["cut_off_rate"] = _to_num(value)
        elif "средневзвешенная ставка" in label:
            current["avg_rate"] = _to_num(value)
        elif "срок" in label and "дни" in label:
            term_val = _to_num(value)
            current["term"] = int(term_val) if pd.notna(term_val) else None

    if current and current.get("demand") is not None:
        records.append(current)

    for rec in records:
        rec["date"] = date
    return records


# добавляю cover_ratio и спред к ключевой ставке
def _finalize_auction_df(df_out, start, end):
    df_out = df_out.drop_duplicates(subset=["date", "demand"])
    df_out["date"] = pd.to_datetime(df_out["date"], dayfirst=True).dt.normalize().astype("datetime64[us]")

    key_rate = fetch_key_rate(start=start, end=end)
    key_rate["date"] = pd.to_datetime(key_rate["date"], dayfirst=True).dt.normalize().astype("datetime64[us]")

    df_out = df_out.sort_values("date")
    key_rate = key_rate.sort_values("date")

    df_out["cover_ratio"] = df_out["demand"] / df_out["placement"]
    df_out = pd.merge_asof(df_out, key_rate, on="date", direction="backward",
                           allow_exact_matches=True)

    rate_col = "avg_rate" if "avg_rate" in df_out.columns else "cut_off_rate"
    if rate_col in df_out.columns:
        df_out["spread_to_key"] = df_out[rate_col] - df_out["key_rate"]
    else:
        df_out["spread_to_key"] = float("nan")

    return df_out.sort_values("date").reset_index(drop=True)


# конвертирую строку в число — убираю пробелы и заменяю запятую
def _to_num(val):
    try:
        return float(str(val).replace(" ", "").replace(",", "."))
    except:
        return float("nan")


# синтетика М2 — использую если ЦБ недоступен
def _synthetic_m2() -> pd.DataFrame:
    np.random.seed(43)
    dates = pd.bdate_range(DATE_START, DATE_END, freq="W-TUE")
    n = len(dates)

    key_rate = _synthetic_keyrate(dates)
    demand = np.random.uniform(300, 900, n)
    placement = demand * np.random.uniform(0.5, 1.0, n)
    cover = demand / placement
    cutoff = key_rate + np.random.normal(0.05, 0.1, n)
    rate_spread = cutoff - key_rate

    df = pd.DataFrame({
        "date": dates,
        "demand": demand,
        "placement": placement,
        "cutoff_rate": cutoff,
        "key_rate": key_rate,
        "cover_ratio": cover,
        "rate_spread": rate_spread
    })
    df = df.set_index("date")
    df["cover_ratio"] = _inject_stress(df["cover_ratio"], STRESS_DATES, level=2.5, width=15)
    df["rate_spread"] = _inject_stress(df["rate_spread"], STRESS_DATES, level=3.0, width=15)
    return df.reset_index()


# исторические значения ключевой ставки ЦБ — ступенчатый график
def _synthetic_keyrate(dates=None) -> np.ndarray:
    if dates is None:
        dates = pd.bdate_range(DATE_START, DATE_END)
    kr_schedule = [
        ("2014-01-01", 5.5), ("2014-03-03", 7.0), ("2014-04-28", 7.5),
        ("2014-07-28", 8.0), ("2014-11-05", 9.5), ("2014-12-12", 10.5),
        ("2014-12-16", 17.0), ("2015-02-02", 15.0), ("2015-03-16", 14.0),
        ("2015-05-05", 12.5), ("2015-06-16", 11.5), ("2015-08-03", 11.0),
        ("2015-10-30", 11.0), ("2016-06-14", 10.5), ("2016-09-19", 10.0),
        ("2017-03-27", 9.75), ("2017-05-02", 9.25), ("2017-06-19", 9.0),
        ("2017-09-18", 8.5), ("2017-10-30", 8.25), ("2017-12-18", 7.75),
        ("2018-02-09", 7.5), ("2018-03-26", 7.25), ("2018-09-14", 7.5),
        ("2018-12-17", 7.75), ("2019-06-17", 7.5), ("2019-07-29", 7.25),
        ("2019-09-09", 7.0), ("2019-10-28", 6.5), ("2019-12-16", 6.25),
        ("2020-02-10", 6.0), ("2020-04-27", 5.5), ("2020-06-22", 4.5),
        ("2020-07-27", 4.25), ("2021-03-22", 4.5), ("2021-04-26", 5.0),
        ("2021-06-11", 5.5), ("2021-07-23", 6.5), ("2021-09-10", 6.75),
        ("2021-10-22", 7.5), ("2021-12-17", 8.5), ("2022-02-28", 20.0),
        ("2022-04-11", 17.0), ("2022-05-04", 14.0), ("2022-05-27", 11.0),
        ("2022-06-10", 9.5), ("2022-07-25", 8.0), ("2022-09-16", 7.5),
        ("2023-07-21", 8.5), ("2023-08-15", 12.0), ("2023-09-15", 13.0),
        ("2023-10-27", 15.0), ("2023-12-15", 16.0), ("2024-12-31", 21.0),
    ]
    dates_ts = pd.DatetimeIndex(dates)
    rates = np.full(len(dates_ts), 5.5)
    for d, v in kr_schedule:
        rates[dates_ts >= d] = v
    return rates


def get_excel_links_from_auction_page():
    url = "https://minfin.gov.ru/ru/perfomance/public_debt/internal/operations/ofz/auction"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"  [М3] Ошибка подключения: {e}")
        return []

    pattern = r'href="([^"]*\.xlsx?)"'
    excel_links = re.findall(pattern, response.text, re.IGNORECASE)

    full_links = []
    seen_urls = set()

    for link in excel_links:
        if link.startswith('/'):
            full_url = urljoin("https://minfin.gov.ru", link)
        else:
            full_url = link

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        filename = link.split('/')[-1]
        year_match = re.search(r'(\d{4})', filename)
        year = year_match.group(1) if year_match else "unknown"

        full_links.append({
            'url': full_url,
            'filename': filename,
            'year': year
        })

    return full_links


def download_new_auction_files(data_dir="data", force=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_dir)
    os.makedirs(data_path, exist_ok=True)

    print("[М3] Поиск файлов аукционов на сайте Минфина")

    links = get_excel_links_from_auction_page()

    if not links:
        print("[М3]  Не найдено ссылок на Excel-файлы")
        return []

    print(f"[М3] Найдено файлов: {len(links)}")
    for link in links:
        print(f"    - {link['filename']} ({link['year']})")

    existing_files = set()
    if os.path.exists(data_path):
        for f in os.listdir(data_path):
            if f.startswith("INTERNET_Auction_Results_rus_"):
                existing_files.add(f)

    print(f"\n[М3] Уже есть в папке: {len(existing_files)} файлов")

    downloaded = []
    for link in links:
        if not force and link['filename'] in existing_files:
            print(f"   Пропускаем: {link['filename']}")
            continue

        target_path = os.path.join(data_path, link['filename'])
        print(f"   Скачиваем: {link['filename']}")

        try:
            file_resp = requests.get(link['url'], headers=HEADERS, timeout=60)
            file_resp.raise_for_status()

            with open(target_path, 'wb') as f:
                f.write(file_resp.content)

            try:
                if target_path.endswith('.xls'):
                    pd.read_excel(target_path, engine='xlrd', nrows=1)
                else:
                    pd.read_excel(target_path, nrows=1)
                downloaded.append(link['filename'])
                print(f"       Успешно")
            except Exception as e:
                print(f"       Файл повреждён: {e}")
                if os.path.exists(target_path):
                    os.remove(target_path)
        except Exception as e:
            print(f"       Ошибка скачивания: {e}")

    print(f"\n[М3] Загружено новых: {len(downloaded)}")
    return downloaded


def fetch_m3_ofz(start_date="2014-01-01", end_date=None) -> pd.DataFrame:
    download_new_auction_files("data")

    excel_files = []

    for filepath in glob.glob("data/INTERNET_Auction_Results_rus_*.xlsx"):
        excel_files.append(filepath)
    for filepath in glob.glob("data/INTERNET_Auction_Results_rus_*.xls"):
        excel_files.append(filepath)

    for filepath in glob.glob("data/svodnaya_rus_*.xlsx"):
        excel_files.append(filepath)

    excel_files = sorted(set(excel_files))

    if not excel_files:
        raise Exception("Не найдено файлов ОФЗ в папке data/.")

    all_auctions = []

    print("\nЗагрузка данных аукционов ОФЗ из Excel-файлов...")

    for filepath in excel_files:
        year = "unknown"
        try:
            match = re.search(r'(\d{4})', filepath)
            if match:
                year = match.group(1)
        except:
            pass

        try:
            if filepath.endswith('.xls'):
                df_raw = pd.read_excel(filepath, sheet_name=0, header=None, engine='xlrd')
            else:
                df_raw = pd.read_excel(filepath, sheet_name=0, header=None)

            df_raw_str = df_raw.astype(str)

            header_row = None
            for i in range(min(30, len(df_raw))):
                row_values = df_raw_str.iloc[i].str.lower()
                row_text = ' '.join(row_values.values)

                if 'дата аукциона' in row_text or ('дата' in row_text and 'объем' in row_text):
                    header_row = i
                    break

            if header_row is None:
                print(f"  {year}: не найдена строка с заголовками - {os.path.basename(filepath)}")
                continue

            new_columns = []
            for col in df_raw.iloc[header_row]:
                if pd.isna(col):
                    new_columns.append(f"col_{len(new_columns)}")
                else:
                    new_columns.append(str(col).strip())

            df_raw.columns = new_columns
            df = df_raw.iloc[header_row + 1:].copy()
            df = df.reset_index(drop=True)

            col_map = {col: col.lower() for col in df.columns}

            date_col = None
            offer_col = None
            demand_col = None
            yield_col = None
            format_col = None

            for col, col_lower in col_map.items():
                if 'дата аукциона' in col_lower or col_lower == 'дата':
                    date_col = col
                elif 'объем предложения' in col_lower or 'объём предложения' in col_lower:
                    offer_col = col
                elif 'совокупный объем спроса' in col_lower or 'объем спроса' in col_lower:
                    demand_col = col
                elif 'формат' in col_lower:
                    format_col = col
                elif 'доходность по средневзвешенной цене' in col_lower:
                    yield_col = col
                elif 'доходность по средневзве- шенной цене' in col_lower:
                    yield_col = col
                elif 'доходность по средневзвешенной цене*' in col_lower:
                    yield_col = col
                elif 'доходность по средневзве- шенной цене*' in col_lower:
                    yield_col = col
                elif 'доходность по средневзвеш' in col_lower:
                    yield_col = col

            if not all([date_col, offer_col, demand_col]):
                print(f"  {year}: не найдены обязательные колонки (дата={date_col}, offer={offer_col}, demand={demand_col}) - {os.path.basename(filepath)}")
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            df[offer_col] = pd.to_numeric(df[offer_col], errors='coerce')
            df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')

            df = df.dropna(subset=[date_col, offer_col, demand_col])
            df = df[df[date_col].dt.year > 2000]

            if len(df) == 0:
                print(f"  {year}: нет данных после очистки - {os.path.basename(filepath)}")
                continue

            df_result = pd.DataFrame()
            df_result['date'] = df[date_col]
            df_result['offer_vol'] = df[offer_col]
            df_result['demand_vol'] = df[demand_col]

            if yield_col:
                df_result['avg_yield'] = pd.to_numeric(df[yield_col], errors='coerce')
                print(f"  {year}: загружено {len(df_result)} аукционов, доходность из колонки '{yield_col}' - {os.path.basename(filepath)}")
            else:
                df_result['avg_yield'] = np.nan
                print(f"  {year}: загружено {len(df_result)} аукционов, колонка доходности не найдена - {os.path.basename(filepath)}")
                if str(year) == "2014":
                    print(f"\n  Доступные колонки в файле {year}:")
                    for i, col in enumerate(df.columns):
                        print(f"    {i}: '{col}'")

            if format_col:
                format_filter = df[format_col].astype(str).str.lower()
                df_result = df_result[format_filter == 'аукцион']

            df_result = df_result[(df_result['offer_vol'] > 0) & (df_result['demand_vol'] > 0)]

            if len(df_result) > 0:
                all_auctions.append(df_result)
                print(f"  {year}: добавлено {len(df_result)} аукционов после фильтрации")
            else:
                print(f"  {year}: нет данных после фильтрации по объемам")

        except Exception as e:
            print(f"  {year}: ошибка - {str(e)} - {os.path.basename(filepath)}")
            continue

    if not all_auctions:
        raise Exception("Не удалось загрузить данные аукционов ОФЗ. Проверьте наличие файлов в папке data/")

    result = pd.concat(all_auctions, ignore_index=True)
    result = result.sort_values('date').reset_index(drop=True)
    result['date'] = pd.to_datetime(result['date']).dt.normalize()

    result['cover_ratio'] = result['demand_vol'] / result['offer_vol']
    result['cover_ratio'] = result['cover_ratio'].clip(lower=0, upper=10)

    low_quantile = result['cover_ratio'].quantile(0.2)
    high_quantile = result['cover_ratio'].quantile(0.8)
    result['flag_nedospros'] = (result['cover_ratio'] < low_quantile).astype(int)
    result['flag_perespros'] = (result['cover_ratio'] > high_quantile).astype(int)

    if 'avg_yield' in result.columns and result['avg_yield'].notna().any():
        df_with_yield = result.dropna(subset=['avg_yield']).copy()
        df_with_yield = df_with_yield.sort_values('date')

        if len(df_with_yield) >= 10:
            df_with_yield['yield_median'] = (
                df_with_yield['avg_yield']
                .expanding(min_periods=5)
                .median()
            )
            df_with_yield['yield_median'] = df_with_yield['yield_median'].shift(1)
            df_with_yield['yield_median'] = df_with_yield['yield_median'].fillna(df_with_yield['avg_yield'].iloc[0])
            df_with_yield['yield_spread'] = df_with_yield['avg_yield'] - df_with_yield['yield_median']

            result = result.merge(df_with_yield[['date', 'yield_spread']], on='date', how='left')
            result['yield_spread'] = result['yield_spread'].fillna(0)
        else:
            result['yield_spread'] = 0
    else:
        result['yield_spread'] = 0

    print(f"\nМ3: загружено {len(result)} аукционов ОФЗ")
    print(f"   Период: {result['date'].min().date()} - {result['date'].max().date()}")
    print(f"   Cover ratio: средний = {result['cover_ratio'].mean():.2f}, медиана = {result['cover_ratio'].median():.2f}")

    if 'avg_yield' in result.columns and result['avg_yield'].notna().any():
        print(f"   Доходность: средняя = {result['avg_yield'].mean():.2f}%, медиана = {result['avg_yield'].median():.2f}%")

    return result[['date', 'offer_vol', 'demand_vol', 'cover_ratio',
                   'avg_yield', 'yield_spread', 'flag_nedospros', 'flag_perespros']]


# М4: строю налоговый календарь по датам из НК РФ — парсить ФНС не нужно,
# даты детерминированы: ЕНП — 28-е каждого месяца (ст. 431), НДС — 25-е квартала (ст. 174)

def fetch_m4_tax_calendar(start=DATE_START, end=DATE_END) -> pd.DataFrame:
    if end is None:
        end = str(pd.Timestamp.today().normalize())

    bdays = _business_days()
    bdays["date"] = pd.to_datetime(bdays["date"])

    tax_days = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for month in range(1, 13):
            # 28-е — ЕНП: НДС авансы, налог на прибыль, страховые взносы, НДФЛ, акцизы
            tax_days.append(pd.Timestamp(year, month, 28))
            # 25-е — окончательный НДС по итогам квартала
            if month in [3, 6, 9, 12]:
                tax_days.append(pd.Timestamp(year, month, 25))

    tax_days = pd.DatetimeIndex(tax_days)

    bdays["tax_week_flag"] = 0
    bdays["end_of_month_flag"] = 0
    bdays["end_of_quarter_flag"] = 0
    bdays["seasonal_factor"] = 1.0

    # налоговая неделя: ±5 дней до и +3 дня после даты платежа
    for td in tax_days:
        mask = (bdays["date"] >= td - pd.Timedelta(days=5)) & \
               (bdays["date"] <= td + pd.Timedelta(days=3))
        bdays.loc[mask, "tax_week_flag"] = 1

    # конец месяца — последние 3 рабочих дня
    bdays["month"] = bdays["date"].dt.month
    bdays["year"] = bdays["date"].dt.year
    for (y, m), grp in bdays.groupby(["year", "month"]):
        idx = grp.index[-3:]
        bdays.loc[idx, "end_of_month_flag"] = 1

    # конец квартала — последние 5 рабочих дней квартального месяца
    q_months = [3, 6, 9, 12]
    for (y, m), grp in bdays.groupby(["year", "month"]):
        if m in q_months:
            idx = grp.index[-5:]
            bdays.loc[idx, "end_of_quarter_flag"] = 1

    # seasonal_factor применяю как мультипликатор, не суммирую — иначе двойной счёт с М1/М2/М5
    bdays["seasonal_factor"] = (
        1.0
        + 0.15 * bdays["tax_week_flag"]
        + 0.10 * bdays["end_of_month_flag"]
        + 0.15 * bdays["end_of_quarter_flag"]
    ).clip(1.0, 1.4)

    tax_pct = bdays["tax_week_flag"].mean() * 100
    print(f"М4: сформирован календарь, {int(bdays['tax_week_flag'].sum())} налоговых дней "
          f"({tax_pct:.1f}% от всех рабочих дней), "
          f"seasonal_factor: {bdays['seasonal_factor'].min():.2f}–{bdays['seasonal_factor'].max():.2f}")

    return bdays[["date", "tax_week_flag", "end_of_month_flag",
                  "end_of_quarter_flag", "seasonal_factor"]].reset_index(drop=True)


# М5: загружаю остатки бюджетных средств на счетах банков из раздела SORS на сайте ЦБ

def fetch_budget_on_accounts(start_date=None, end_date=None) -> pd.DataFrame:
    # получаю HTML страницы со статистикой
    resp = requests.get(M5_SORS_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # ищу ссылку на Excel внутри блока с бюджетными средствами
    pattern_block = r'id="DropDown2_content".*?href="([^"]+\.xlsx)"'
    match = re.search(pattern_block, html, re.DOTALL)
    if match:
        relative_url = match.group(1)
    else:
        # запасной вариант — любая .xlsx со словом budget
        all_xlsx = re.findall(r'href="([^"]+\.xlsx)"', html)
        budget_links = [l for l in all_xlsx if "budget" in l.lower()]
        if not budget_links:
            raise ValueError("не нашёл ссылку на файл бюджетных средств")
        relative_url = budget_links[0]

    file_url = "https://www.cbr.ru" + relative_url if relative_url.startswith("/") else relative_url
    print(f"М5-SORS: скачиваю {file_url}")

    resp = requests.get(file_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    xls = pd.ExcelFile(BytesIO(resp.content))
    sheet_name = "итого" if "итого" in xls.sheet_names else xls.sheet_names[0]
    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    # структура листа: строка 1 — даты, строка 3 — федбюджет, строка 6 — внебюджетные фонды
    dates = pd.to_datetime(raw.iloc[1, 1:], dayfirst=True, errors="coerce")
    federal_budget = pd.to_numeric(raw.iloc[3, 1:], errors="coerce")
    extrabudget_funds = pd.to_numeric(raw.iloc[6, 1:], errors="coerce")

    df = pd.DataFrame({
        "date": dates,
        "federal_budget": federal_budget,
        "extrabudget_funds": extrabudget_funds,
    })

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    print(f"М5-SORS: загружено {len(df)} записей, "
          f"{df['date'].min().date()} – {df['date'].max().date()}")
    return df.reset_index(drop=True)


# М5: загружаю размещения ЕКС на депозитах с сайта Росказны — данные в XML-файлах

def fetch_roskazna_eks(start_date=None, end_date=None) -> pd.DataFrame:
    import urllib3
    import xml.etree.ElementTree as ET
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Росказна использует российский УЦ — запрашиваю с verify=False
    resp = requests.get(M5_ROSKAZNA_URL, headers=HEADERS, timeout=20, verify=False)
    resp.raise_for_status()
    html = resp.text

    # собираю все ссылки на XML-файлы аукционов
    soup = BeautifulSoup(html, "html.parser")
    xml_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".xml") and "operation-day-files" in href:
            xml_links.append(href)

    if not xml_links:
        raise ValueError("не нашёл XML-файлов на странице Росказны")

    print(f"М5-Росказна: найдено {len(xml_links)} XML-файлов, парсю")

    # парсю каждый XML — извлекаю дату, объём, ставку и количество банков
    records = []
    for href in xml_links:
        url = "https://roskazna.gov.ru" + href if href.startswith("/") else href
        try:
            r = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            # в одном файле может быть несколько аукционов (Depoauc1, Depoauc2, ...)
            for auction in root:
                aucdate = auction.findtext("aucdate")
                maxvol = auction.findtext("maxvol")
                warate = auction.findtext("waacceptrate")
                crbidders = auction.findtext("crbidders")
                acceptcrbidders = auction.findtext("acceptcrbidders")
                if aucdate and maxvol:
                    records.append({
                        "date": pd.to_datetime(aucdate, dayfirst=True, errors="coerce"),
                        "eks_volume_raw": _to_num(maxvol),
                        "eks_rate": _to_num(warate) if warate else float("nan"),
                        "banks_total": _to_num(crbidders) if crbidders else float("nan"),
                        "banks_accepted": _to_num(acceptcrbidders) if acceptcrbidders else float("nan"),
                    })
        except Exception:
            continue

    if not records:
        raise ValueError("не удалось распарсить ни один XML-файл Росказны")

    df = pd.DataFrame(records).dropna(subset=["date"])
    # суммирую объёмы по дате, переводю млн → млрд
    df_out = df.groupby("date").agg(
        eks_volume=("eks_volume_raw", "sum"),
        eks_rate=("eks_rate", "mean"),
        banks_total=("banks_total", "sum"),
        banks_accepted=("banks_accepted", "sum"),
    ).reset_index()
    df_out["eks_volume"] = df_out["eks_volume"] / 1000

    df_out = df_out.sort_values("date").reset_index(drop=True)

    if start_date is not None:
        df_out = df_out[df_out["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_out = df_out[df_out["date"] <= pd.to_datetime(end_date)]

    print(f"М5-Росказна: загружено {len(df_out)} записей, "
          f"{df_out['date'].min().date()} – {df_out['date'].max().date()}")
    return df_out.reset_index(drop=True)


# М5: объединяю все три источника в один датафрейм

def fetch_m5_treasury(start=DATE_START, end=DATE_END):
    if end is None:
        end = pd.Timestamp.today().normalize()

    try:
        # загружаю основную метрику — структурный баланс ликвидности
        r = requests.get(M5_BLIQUIDITY, params=_build_cbr_params(start, end), headers=HEADERS, timeout=20)
        r.raise_for_status()

        tables = pd.read_html(StringIO(r.text), decimal=",", thousands=" ")
        if not tables:
            raise ValueError("таблиц не найдено")
        df = tables[0]

        idx_structural = _find_col(df, "профицит", "ликвидности")
        idx_corr = _find_col(df, "средства", "корсчет")
        idx_oblg = _find_col(df, "обязательства", "итого") or _find_col(df, "9 =")

        df_out = pd.DataFrame({
            "date": df.iloc[:, 0],
            "structural_balance": df.iloc[:, idx_structural],
            "corr_accounts": df.iloc[:, idx_corr],
        })
        if idx_oblg is not None:
            df_out["cb_obligations"] = df.iloc[:, idx_oblg]

        num_cols = ["structural_balance", "corr_accounts"] + \
                   (["cb_obligations"] if "cb_obligations" in df_out.columns else [])
        for c in num_cols:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

        df_out["date"] = pd.to_datetime(df_out["date"], dayfirst=True, errors="coerce")
        df_out = df_out.dropna(subset=["date", "structural_balance"]).sort_values("date")
        df_out["delta_weekly"] = df_out["structural_balance"].diff(5)
        df_out["delta_monthly"] = df_out["structural_balance"].diff(21)
        print("М5-bliquidity: данные загружены с сайта ЦБ")

        # подгружаю второй источник — SORS, данные месячные поэтому merge_asof
        try:
            df_sors = fetch_budget_on_accounts(start_date=start, end_date=end)
            df_out = pd.merge_asof(
                df_out.sort_values("date"),
                df_sors[["date", "federal_budget", "extrabudget_funds"]].sort_values("date"),
                on="date",
                direction="backward"
            )
        except Exception as e:
            print(f"М5-SORS: не загрузилось ({e}), federal_budget = NaN")
            df_out["federal_budget"] = float("nan")
            df_out["extrabudget_funds"] = float("nan")

        # подгружаю третий источник — Росказна
        try:
            df_eks = fetch_roskazna_eks(start_date=start, end_date=end)
            df_out = pd.merge_asof(
                df_out.sort_values("date"),
                df_eks[["date", "eks_volume", "eks_rate", "banks_total", "banks_accepted"]].sort_values("date"),
                on="date",
                direction="backward"
            )
        except Exception as e:
            print(f"М5-Росказна: не загрузилось ({e}), eks_volume = NaN")
            df_out["eks_volume"] = float("nan")
            df_out["eks_rate"] = float("nan")
            df_out["banks_total"] = float("nan")
            df_out["banks_accepted"] = float("nan")

        return df_out.reset_index(drop=True)

    except Exception as e:
        print(f"[M5-ERR] {type(e).__name__}: {e}")
        print("М5: ЦБ недоступен, использую синтетические данные")
        return _synthetic_m5()


# синтетика М5 — все три источника, использую если ЦБ недоступен
def _synthetic_m5() -> pd.DataFrame:
    np.random.seed(45)
    bdays = _business_days()
    n = len(bdays)

    structural = -1.5 + np.cumsum(np.random.normal(0, 0.05, n))
    structural = np.clip(structural, -6, 2)

    federal_budget = 1200 + np.cumsum(np.random.normal(0, 20, n))
    federal_budget = np.clip(federal_budget, 200, 4000)
    extrabudget_funds = 400 + np.cumsum(np.random.normal(0, 8, n))
    extrabudget_funds = np.clip(extrabudget_funds, 50, 1200)

    eks_volume = 800 + np.cumsum(np.random.normal(0, 15, n))
    eks_volume = np.clip(eks_volume, 100, 3000)

    df = bdays.copy()
    df["structural_balance"] = structural
    df["federal_budget"] = federal_budget
    df["extrabudget_funds"] = extrabudget_funds
    df["eks_volume"] = eks_volume
    df["eks_rate"] = float("nan")

    df = df.set_index("date")
    # в стресс баланс и бюджет падают
    df["structural_balance"] = _inject_stress(df["structural_balance"], STRESS_DATES, level=-3.0, width=20)
    df["federal_budget"] = _inject_stress(df["federal_budget"], STRESS_DATES, level=-400, width=20)
    df["eks_volume"] = _inject_stress(df["eks_volume"], STRESS_DATES, level=-300, width=20)
    df = df.reset_index()
    df["delta_weekly"] = df["structural_balance"].diff(5)
    df["delta_monthly"] = df["structural_balance"].diff(21)
    df["banks_total"] = float("nan")
    df["banks_accepted"] = float("nan")
    return df


# запускаю загрузку всех модулей
def load_all_data() -> dict:
    print("=" * 50)
    print("загружаю данные для всех модулей")
    data = {
        "m1": fetch_m1_reserves(),
        "m2": fetch_m2_repo(),
        "m3": fetch_m3_ofz(),
        "m4": fetch_m4_tax_calendar(),
        "m5": fetch_m5_treasury(),
    }
    for k, v in data.items():
        print(f"{k.upper()}: {len(v)} записей, {v['date'].min().date()} – {v['date'].max().date()}")
    return data


if __name__ == "__main__":
    data = load_all_data()
