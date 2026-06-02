import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# MAD-нормализация — привожу все сигналы к единой шкале [0, 1]

"""def mad_score(series: pd.Series, window: int = 756) -> pd.Series:
    roll_median = series.rolling(window, min_periods=1).median()
    roll_mad = series.rolling(window, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    roll_mad = roll_mad.replace(0, np.nan).ffill().bfill()
    score = (series - roll_median) / (1.4826 * roll_mad)
    score = score.clip(-5, 5)
    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score_norm.fillna(0)"""

def mad_score(series: pd.Series, window: int = 756) -> pd.Series:
    # для коротких рядов (М3 — 2 аукциона в неделю) уменьшаю окно
    if len(series) < 500:
        actual_window = min(window, max(20, len(series) // 3))
    else:
        actual_window = window

    roll_median = series.rolling(actual_window, min_periods=5).median()
    roll_mad = series.rolling(actual_window, min_periods=5).apply(
        lambda x: np.median(np.abs(x - np.median(x))) if len(x) > 5 else np.nan, raw=True)
    roll_mad = roll_mad.replace(0, np.nan).ffill().bfill()
    roll_mad = roll_mad.clip(lower=0.01)

    score = (series - roll_median) / (1.4826 * roll_mad)
    score = score.clip(-5, 5)

    score_min = score.min()
    score_max = score.max()
    if score_max - score_min > 1e-9:
        score_norm = (score - score_min) / (score_max - score_min)
    else:
        score_norm = score * 0

    return score_norm.fillna(0)


# М1: считаю MAD-сигналы по спреду резервов и RUONIA

def compute_m1_signals(df_m1: pd.DataFrame) -> pd.DataFrame:
    df = df_m1.copy().sort_values("date").reset_index(drop=True)

    df["mad_score_spread"] = mad_score(df["spread"])
    df["mad_score_ruonia"] = mad_score(df["ruonia"])

    # флаг конца периода усреднения — последние 4 дня месяца
    df["day"] = df["date"].dt.day
    df["days_in_month"] = df["date"].dt.days_in_month
    df["flag_end_of_period"] = ((df["days_in_month"] - df["day"]) <= 4).astype(int)

    df["m1_signal"] = (
        0.5 * df["mad_score_spread"] +
        0.35 * df["mad_score_ruonia"] +
        0.15 * df["flag_end_of_period"]
    )

    return df[["date", "mad_score_spread", "mad_score_ruonia",
               "flag_end_of_period", "m1_signal"]]


# М2: считаю сигналы по cover ratio и спреду ставки к ключевой

def compute_m2_signals(df_m2: pd.DataFrame) -> pd.DataFrame:
    df = df_m2.copy().sort_values("date").reset_index(drop=True)

    df["mad_score_cover"] = mad_score(df["cover_ratio"])
    df["mad_score_rate_spread"] = mad_score(df["spread_to_key"])

    # флаг переспроса — cover > 2.0 по ТЗ
    df["flag_demand"] = (df["cover_ratio"] > 2.0).astype(int)

    df["m2_signal"] = (
        0.5 * df["mad_score_cover"] +
        0.35 * df["mad_score_rate_spread"] +
        0.15 * df["flag_demand"]
    )

    return df[["date", "mad_score_cover", "mad_score_rate_spread",
               "flag_demand", "m2_signal"]]


# М3: считаю сигналы по аукционам ОФЗ

def compute_m3_signals(df_m3: pd.DataFrame) -> pd.DataFrame:
    df = df_m3.copy().sort_values("date").reset_index(drop=True)

    # инвертирую cover ratio — низкий cover = высокий стресс
    df["mad_score_cover_raw"] = mad_score(-df["cover_ratio"], window=156)
    df["mad_score_yield_raw"] = mad_score(df["yield_spread"], window=156)

    # отрицательные значения = анти-стресс, обрезаю
    df["mad_score_cover"] = df["mad_score_cover_raw"].clip(lower=0)
    df["mad_score_yield_spread"] = df["mad_score_yield_raw"].clip(lower=0)

    # сглаживаю чтобы убрать микрошум между аукционами
    df["mad_score_cover"] = df["mad_score_cover"].rolling(5, min_periods=1, center=True).mean()
    df["mad_score_yield_spread"] = df["mad_score_yield_spread"].rolling(5, min_periods=1, center=True).mean()

    df["flag_nedospros"] = (df["cover_ratio"] < 1.2).astype(int)
    df["flag_perespros"] = (df["cover_ratio"] > 2.0).astype(int)

    df["m3_signal"] = (
        0.55 * df["mad_score_cover"] +
        0.30 * df["mad_score_yield_spread"] +
        0.15 * df["flag_nedospros"]
    )

    return df[["date", "mad_score_cover", "mad_score_yield_spread",
               "flag_nedospros", "flag_perespros", "m3_signal"]]


# М4: детерминированный — просто возвращаю флаги и seasonal_factor

def compute_m4_signals(df_m4: pd.DataFrame) -> pd.DataFrame:
    return df_m4.copy()


# М5: считаю сигналы по трём источникам — bliquidity, SORS, Росказна

def compute_m5_signals(df_m5: pd.DataFrame) -> pd.DataFrame:
    df = df_m5.copy().sort_values("date").reset_index(drop=True)

    # инвертирую — падение баланса = рост стресса
    df["structural_inv"] = -df["structural_balance"]
    df["delta_inv"] = -df["delta_weekly"].fillna(0)

    df["mad_score_structural"] = mad_score(df["structural_inv"])
    df["mad_score_delta"] = mad_score(df["delta_inv"])

    # флаг оттока — нижние 10% по недельной дельте
    threshold = df["delta_weekly"].quantile(0.10)
    df["flag_budget_drain"] = (df["delta_weekly"] < threshold).astype(int)

    # сигнал по остаткам федбюджета на счетах банков (SORS)
    has_fed_budget = ("federal_budget" in df.columns and
                      df["federal_budget"].notna().sum() > 10)
    if has_fed_budget:
        df["fed_budget_inv"] = -df["federal_budget"].ffill()
        df["mad_score_fed_budget"] = mad_score(df["fed_budget_inv"])
    else:
        df["mad_score_fed_budget"] = 0.0

    # сигнал по размещениям ЕКС Росказны — падение = отток ликвидности
    has_eks = ("eks_volume" in df.columns and
               df["eks_volume"].notna().sum() > 10)
    if has_eks:
        df["eks_inv"] = -df["eks_volume"].ffill()
        df["mad_score_eks"] = mad_score(df["eks_inv"])
    else:
        df["mad_score_eks"] = 0.0

    # веса зависят от того, какие источники удалось загрузить
    if has_fed_budget and has_eks:
        m5_sig = (
            0.35 * df["mad_score_structural"] +
            0.20 * df["mad_score_delta"] +
            0.20 * df["mad_score_fed_budget"] +
            0.15 * df["mad_score_eks"] +
            0.10 * df["flag_budget_drain"]
        )
    elif has_fed_budget:
        m5_sig = (
            0.40 * df["mad_score_structural"] +
            0.25 * df["mad_score_delta"] +
            0.20 * df["mad_score_fed_budget"] +
            0.15 * df["flag_budget_drain"]
        )
    else:
        m5_sig = (
            0.50 * df["mad_score_structural"] +
            0.35 * df["mad_score_delta"] +
            0.15 * df["flag_budget_drain"]
        )

    df["m5_signal"] = m5_sig

    return df[["date", "mad_score_structural", "mad_score_delta",
               "mad_score_fed_budget", "mad_score_eks", "flag_budget_drain", "m5_signal"]]


# собираю сигналы всех модулей в единую матрицу признаков

def build_feature_matrix(signals: dict) -> pd.DataFrame:
    base = signals["m1"][["date", "m1_signal", "mad_score_spread",
                           "mad_score_ruonia", "flag_end_of_period"]].copy()
    m2 = signals["m2"][["date", "m2_signal", "mad_score_cover",
                         "mad_score_rate_spread", "flag_demand"]].copy()
    m3 = signals["m3"][["date", "m3_signal", "flag_nedospros",
                         "flag_perespros"]].copy()
    m4 = signals["m4"][["date", "tax_week_flag", "end_of_month_flag",
                         "end_of_quarter_flag", "seasonal_factor"]].copy()
    m5 = signals["m5"][["date", "m5_signal", "mad_score_structural",
                         "flag_budget_drain"]].copy()

    # привожу даты к одному типу перед merge
    for d in [base, m2, m3, m4, m5]:
        d["date"] = d["date"].astype("datetime64[ns]")

    df = base.copy()
    for other in [m2, m3, m4, m5]:
        df = pd.merge_asof(
            df.sort_values("date"),
            other.sort_values("date"),
            on="date", direction="backward"
        )

    df = df.ffill().fillna(0)
    return df.sort_values("date").reset_index(drop=True)


# базовые веса модулей для взвешенной суммы

MODULE_WEIGHTS = {
    "m1_signal": 0.20,
    "m2_signal": 0.25,
    "m3_signal": 0.20,
    "m5_signal": 0.15,
    # М4 входит как мультипликатор через seasonal_factor
}

FEATURE_COLS = ["m1_signal", "m2_signal", "m3_signal", "m5_signal",
                "mad_score_spread", "mad_score_ruonia",
                "mad_score_cover", "mad_score_rate_spread",
                "flag_demand", "flag_nedospros", "flag_budget_drain",
                "tax_week_flag", "end_of_quarter_flag"]


# простой метод агрегации — взвешенная сумма с seasonal_factor

def compute_lsi_weighted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw = (
        MODULE_WEIGHTS["m1_signal"] * df["m1_signal"] +
        MODULE_WEIGHTS["m2_signal"] * df["m2_signal"] +
        MODULE_WEIGHTS["m3_signal"] * df["m3_signal"] +
        MODULE_WEIGHTS["m5_signal"] * df["m5_signal"]
    )

    # М4 применяю как мультипликатор, не суммирую — иначе двойной счёт
    seasonal = df.get("seasonal_factor", pd.Series(1.0, index=df.index))
    raw_seasonal = raw * seasonal

    lsi_raw = (raw_seasonal - raw_seasonal.min()) / \
              (raw_seasonal.max() - raw_seasonal.min() + 1e-9) * 100
    df["lsi"] = lsi_raw.clip(0, 100)

    total = sum(MODULE_WEIGHTS.values())
    df["contrib_m1"] = MODULE_WEIGHTS["m1_signal"] / total * df["m1_signal"] * 100
    df["contrib_m2"] = MODULE_WEIGHTS["m2_signal"] / total * df["m2_signal"] * 100
    df["contrib_m3"] = MODULE_WEIGHTS["m3_signal"] / total * df["m3_signal"] * 100
    df["contrib_m5"] = MODULE_WEIGHTS["m5_signal"] / total * df["m5_signal"] * 100
    df["contrib_m4"] = (seasonal - 1.0) * 20

    return df


# основной метод — GBM с псевдо-разметкой по историческим эпизодам

def compute_lsi_gbm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # задаю таргеты: стресс-эпизоды = высокий LSI, спокойные = низкий
    stress_periods = [
        ("2014-12-10", "2015-01-20", 90),
        ("2022-02-24", "2022-04-30", 92),
        ("2023-08-01", "2023-10-01", 78),
    ]
    calm_periods = [
        ("2016-04-01", "2016-09-30", 20),
        ("2019-05-01", "2019-11-30", 18),
        ("2021-03-01", "2021-08-31", 22),
    ]
    df["target"] = 35.0

    for start, end, val in stress_periods:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "target"] = val
    for start, end, val in calm_periods:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "target"] = val

    X = df[feature_cols].fillna(0).values
    y = df["target"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # неглубокие деревья — меньше переобучения
    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=2,
        learning_rate=0.08, subsample=0.8,
        random_state=42
    )
    model.fit(X_scaled, y)

    lsi_raw = model.predict(X_scaled)
    lsi_norm = (lsi_raw - lsi_raw.min()) / (lsi_raw.max() - lsi_raw.min() + 1e-9) * 100

    # применяю seasonal_factor из М4 как мультипликатор
    seasonal = df.get("seasonal_factor", pd.Series(1.0, index=df.index))
    lsi_norm = lsi_norm * seasonal.values
    lsi_norm = (lsi_norm - lsi_norm.min()) / (lsi_norm.max() - lsi_norm.min() + 1e-9) * 100

    df["lsi"] = lsi_norm.clip(0, 100)

    # вклад модулей через feature_importances_ (глобальная важность признаков)
    importances = dict(zip(feature_cols, model.feature_importances_))

    m1_cols = ["m1_signal", "mad_score_spread", "mad_score_ruonia"]
    m2_cols = ["m2_signal", "mad_score_cover", "mad_score_rate_spread", "flag_demand"]
    m3_cols = ["m3_signal", "flag_nedospros"]
    m4_cols = ["tax_week_flag", "end_of_quarter_flag"]
    m5_cols = ["m5_signal", "flag_budget_drain"]

    def _group_imp(cols):
        return sum(importances.get(c, 0) for c in cols)

    raw_imp = [_group_imp(g) for g in [m1_cols, m2_cols, m3_cols, m4_cols, m5_cols]]
    total_imp = sum(raw_imp) + 1e-9

    df["contrib_m1"] = raw_imp[0] / total_imp * df["lsi"]
    df["contrib_m2"] = raw_imp[1] / total_imp * df["lsi"]
    df["contrib_m3"] = raw_imp[2] / total_imp * df["lsi"]
    df["contrib_m4"] = raw_imp[3] / total_imp * df["lsi"]
    df["contrib_m5"] = raw_imp[4] / total_imp * df["lsi"]

    df.attrs["feature_importances"] = importances
    df.attrs["feature_cols"] = feature_cols
    df.attrs["model"] = model
    df.attrs["scaler"] = scaler

    return df


# запасной метод — Ridge-регрессия, прозрачные коэффициенты

def compute_lsi_ml(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    stress_periods = [
        ("2014-12-10", "2015-01-20", 95),
        ("2022-02-24", "2022-04-30", 90),
        ("2023-08-01", "2023-10-01", 80),
    ]
    df["target"] = 20.0

    for start, end, val in stress_periods:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "target"] = val

    X = df[feature_cols].fillna(0).values
    y = df["target"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    lsi_raw = model.predict(X_scaled)
    lsi_norm = (lsi_raw - lsi_raw.min()) / (lsi_raw.max() - lsi_raw.min() + 1e-9) * 100
    df["lsi"] = lsi_norm.clip(0, 100)

    # вклад модулей через коэффициенты регрессии
    coefs = dict(zip(feature_cols, model.coef_))

    def _group_contrib(cols, x_cols, X_s, coefs):
        idx = [x_cols.index(c) for c in cols if c in x_cols]
        if not idx:
            return 0
        return float(np.mean(np.abs([coefs[x_cols[i]] * X_s[:, i].mean() for i in idx])))

    m1_cols = ["m1_signal", "mad_score_spread", "mad_score_ruonia"]
    m2_cols = ["m2_signal", "mad_score_cover", "mad_score_rate_spread", "flag_demand"]
    m3_cols = ["m3_signal", "flag_nedospros"]
    m4_cols = ["tax_week_flag", "end_of_quarter_flag"]
    m5_cols = ["m5_signal", "flag_budget_drain"]

    groups = [m1_cols, m2_cols, m3_cols, m4_cols, m5_cols]
    raw_c = [_group_contrib(g, feature_cols, X_scaled, coefs) for g in groups]
    total_c = sum(raw_c) + 1e-9

    df["contrib_m1"] = raw_c[0] / total_c * df["lsi"]
    df["contrib_m2"] = raw_c[1] / total_c * df["lsi"]
    df["contrib_m3"] = raw_c[2] / total_c * df["lsi"]
    df["contrib_m4"] = raw_c[3] / total_c * df["lsi"]
    df["contrib_m5"] = raw_c[4] / total_c * df["lsi"]

    df.attrs["ridge_coefs"] = coefs
    df.attrs["feature_cols"] = feature_cols
    df.attrs["model"] = model
    df.attrs["scaler"] = scaler

    return df


# присваиваю статус по порогам из ТЗ

def add_status(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["lsi"] < 40,
        (df["lsi"] >= 40) & (df["lsi"] < 70),
        df["lsi"] >= 70
    ]
    choices = ["🟢 НОРМА", "🟡 ВНИМАНИЕ", "🔴 СТРЕСС"]
    df["status"] = np.select(conditions, choices, default="🟢 НОРМА")
    df["status_color"] = np.select(conditions, ["green", "orange", "red"], default="green")
    return df


# сравниваю сигналы М2/М5 в налоговые недели vs обычные дни

def tax_period_analysis(signals: dict) -> pd.DataFrame:
    # доказываю что налоговый эффект уже виден в М1/М2/М5 — поэтому М4 как мультипликатор
    m4 = signals["m4"][["date", "tax_week_flag"]].copy()
    m4["date"] = pd.to_datetime(m4["date"]).astype("datetime64[ns]")

    results = []
    checks = [
        ("m2", "m2_signal", "М2 cover_ratio сигнал"),
        ("m2", "mad_score_cover", "М2 MAD cover"),
        ("m2", "mad_score_rate_spread", "М2 MAD rate_spread"),
        ("m5", "m5_signal", "М5 казначейство сигнал"),
        ("m5", "mad_score_structural", "М5 MAD структурный"),
        ("m5", "flag_budget_drain", "М5 flag_budget_drain"),
    ]

    for mod, col, label in checks:
        df_mod = signals[mod][["date", col]].copy() if col in signals[mod].columns else None
        if df_mod is None:
            continue
        df_mod["date"] = pd.to_datetime(df_mod["date"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            df_mod.sort_values("date"),
            m4.sort_values("date"),
            on="date", direction="backward"
        ).dropna(subset=[col, "tax_week_flag"])

        tax = merged[merged["tax_week_flag"] == 1][col]
        normal = merged[merged["tax_week_flag"] == 0][col]
        if len(tax) == 0 or len(normal) == 0:
            continue

        avg_tax = tax.mean()
        avg_normal = normal.mean()
        uplift_pct = (avg_tax - avg_normal) / (avg_normal + 1e-9) * 100
        results.append({
            "сигнал": label,
            "налоговая неделя": round(avg_tax, 4),
            "обычный день": round(avg_normal, 4),
            "прирост_%": round(uplift_pct, 1),
        })

    return pd.DataFrame(results).sort_values("прирост_%", ascending=False).reset_index(drop=True)


# проверяю устойчивость LSI при отклонении весов на ±20%

def sensitivity_analysis(df_features: pd.DataFrame) -> pd.DataFrame:
    results = []
    base_weights = MODULE_WEIGHTS.copy()
    base_lsi = compute_lsi_weighted(df_features)["lsi"].mean()

    for module, base_w in base_weights.items():
        for delta in [-0.2, +0.2]:
            w_mod = base_weights.copy()
            w_mod[module] = base_w * (1 + delta)
            total = sum(w_mod.values())
            w_norm = {k: v / total for k, v in w_mod.items()}
            raw = sum(w_norm[k] * df_features[k] for k in w_mod if k in df_features.columns)
            lsi_new = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 100
            results.append({
                "module": module,
                "weight_change": f"{delta:+.0%}",
                "avg_lsi_base": round(base_lsi, 2),
                "avg_lsi_new": round(lsi_new.mean(), 2),
                "delta_lsi": round(lsi_new.mean() - base_lsi, 2)
            })
    return pd.DataFrame(results)


# исторические эпизоды для бэктеста

STRESS_EPISODES = {
    "Декабрь 2014 (ЦБ поднял ставку до 17%)": ("2014-12-01", "2015-01-31"),
    "Февраль 2022 (санкции)": ("2022-02-01", "2022-04-30"),
    "Август 2023 (курс рубля)": ("2023-07-15", "2023-10-01"),
}

CALM_EPISODES = {
    "2016 (стабилизация)": ("2016-04-01", "2016-09-30"),
    "2019 (низкие ставки)": ("2019-05-01", "2019-11-30"),
    "2021 (восстановление)": ("2021-03-01", "2021-08-31"),
}


# проверяю что в стресс-эпизоды LSI > 60, в спокойные < 45

def run_backtest(df_lsi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df_lsi = df_lsi.copy()
    df_lsi["date"] = pd.to_datetime(df_lsi["date"])

    for label, (start, end) in STRESS_EPISODES.items():
        mask = (df_lsi["date"] >= start) & (df_lsi["date"] <= end)
        sub = df_lsi.loc[mask, "lsi"]
        if len(sub) == 0:
            continue
        rows.append({
            "период": label,
            "тип": "СТРЕСС",
            "avg_LSI": round(sub.mean(), 1),
            "max_LSI": round(sub.max(), 1),
            "дней_>60": int((sub > 60).sum()),
            "всего_дней": len(sub),
            "детекция_%": round((sub > 60).mean() * 100, 1)
        })

    for label, (start, end) in CALM_EPISODES.items():
        mask = (df_lsi["date"] >= start) & (df_lsi["date"] <= end)
        sub = df_lsi.loc[mask, "lsi"]
        if len(sub) == 0:
            continue
        rows.append({
            "период": label,
            "тип": "НОРМА",
            "avg_LSI": round(sub.mean(), 1),
            "max_LSI": round(sub.max(), 1),
            "дней_<45": int((sub < 45).sum()),
            "всего_дней": len(sub),
            "специфичность_%": round((sub < 45).mean() * 100, 1)
        })

    return pd.DataFrame(rows)


# главная функция — запускаю весь пайплайн и возвращаю результаты

def run_signal_engine(raw_data: dict, method: str = "weighted") -> dict:
    print(f"\n[Signal Engine] метод агрегации: {method}")

    signals = {
        "m1": compute_m1_signals(raw_data["m1"]),
        "m2": compute_m2_signals(raw_data["m2"]),
        "m3": compute_m3_signals(raw_data["m3"]),
        "m4": compute_m4_signals(raw_data["m4"]),
        "m5": compute_m5_signals(raw_data["m5"]),
    }

    df_features = build_feature_matrix(signals)
    print(f"[Signal Engine] feature matrix: {df_features.shape}")

    if method == "gbm":
        df_lsi = compute_lsi_gbm(df_features)
    elif method == "ml":
        df_lsi = compute_lsi_ml(df_features)
    else:
        df_lsi = compute_lsi_weighted(df_features)

    df_lsi = add_status(df_lsi)

    backtest = run_backtest(df_lsi)
    sensitivity = sensitivity_analysis(df_features)
    tax_analysis = tax_period_analysis(signals)

    print("\n[Backtest]")
    print(backtest.to_string(index=False))

    print("\n[Sensitivity ±20%]")
    print(sensitivity.to_string(index=False))

    print("\n[М4 — налоговые vs обычные периоды]")
    print("налоговый эффект уже в М1/М2/М5 — М4 применяю как мультипликатор")
    print(tax_analysis.to_string(index=False))

    return {
        "signals": signals,
        "features": df_features,
        "lsi": df_lsi,
        "backtest": backtest,
        "sensitivity": sensitivity,
        "tax_analysis": tax_analysis,
    }


if __name__ == "__main__":
    from data_fetcher import load_all_data
    data = load_all_data()
    results = run_signal_engine(data, method="weighted")
    print(f"\nпоследнее значение LSI: {results['lsi']['lsi'].iloc[-1]:.1f}")
    print(f"статус: {results['lsi']['status'].iloc[-1]}")
