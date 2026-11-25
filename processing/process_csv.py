"""CSV 처리: 서울시 상권분석서비스 4종 + 자치구 면적표 → 자치구 × 분기 통합 테이블(data.csv)

입력 (cp949)
  유동인구   : 자치구 × 분기
  점포       : 자치구 × 분기 × 서비스업종(100종) → 자치구 합산, 개업률·폐업률은 점포 수 기준 재계산
  추정매출   : 자치구 × 분기 × 서비스업종(63종)  → 자치구 합산
  소득소비   : 자치구 × 분기 (컬럼명은 '행정동_코드'지만 값은 자치구 코드)
  면적표     : 자치구 면적(km²) (utf-8)

사용: python process_csv.py <유동인구.csv> <점포.csv> <추정매출.csv> <소득소비.csv> <면적.csv> <out_dir>
"""
import sys
from pathlib import Path

import pandas as pd

FOOT, STORE, SALES, SPEND, AREA, OUT = sys.argv[1:7]
OUT = Path(OUT)
OUT.mkdir(parents=True, exist_ok=True)

KEY = ["년분기", "자치구코드", "자치구명"]
TIME = ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]


def read(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def key_cols(df, q="기준_년분기_코드", c="자치구_코드", n="자치구_코드_명"):
    return df.rename(columns={q: "년분기", c: "자치구코드", n: "자치구명"})


log = []

# 1) 유동인구 ─────────────────────────────────────────────
foot = key_cols(read(FOOT))
foot_out = foot[KEY].copy()
foot_out["총유동인구"] = foot["총_유동인구_수"]
for t in TIME:
    foot_out[f"시간대_{t}_유동인구"] = foot[f"시간대_{t}_유동인구_수"]
for d in DAYS:
    foot_out[f"{d}요일_유동인구"] = foot[f"{d}요일_유동인구_수"]
for col in ["남성", "여성"]:
    foot_out[f"{col}_유동인구"] = foot[f"{col}_유동인구_수"]
for a in ["10", "20", "30", "40", "50", "60_이상"]:
    foot_out[f"연령대_{a}_유동인구"] = foot[f"연령대_{a}_유동인구_수"]
log.append(f"유동인구: {foot.shape[0]}행 (자치구 단위, 집계 불필요)")

# 2) 점포 ─────────────────────────────────────────────────
store = key_cols(read(STORE))
n_ind = store["서비스_업종_코드"].nunique()
st = store.groupby(KEY, as_index=False)[["전체_점포_수", "프랜차이즈_점포_수", "개업_점포_수", "폐업_점포_수"]].sum()
st = st.rename(columns={"전체_점포_수": "점포수", "프랜차이즈_점포_수": "프랜차이즈점포수",
                        "개업_점포_수": "개업점포수", "폐업_점포_수": "폐업점포수"})
# 원본 업종별 개업율 = 개업_점포_수 / 전체_점포_수 × 100 → 자치구 단위도 합계 기준으로 재계산
st["개업률"] = (st["개업점포수"] / st["점포수"] * 100).round(2)
st["폐업률"] = (st["폐업점포수"] / st["점포수"] * 100).round(2)
for c in ["점포수", "프랜차이즈점포수", "개업점포수", "폐업점포수"]:
    st[c] = st[c].round().astype("int64")
log.append(f"점포: {store.shape[0]:,}행 (업종 {n_ind}종) → 자치구 합산 {st.shape[0]}행, 개업률·폐업률은 합계 기준 재계산")

# 3) 추정매출 ─────────────────────────────────────────────
sales = key_cols(read(SALES))
n_ind_s = sales["서비스_업종_코드"].nunique()
cols = {"당월_매출_금액": "총추정매출액", "당월_매출_건수": "총추정매출건수",
        "주중_매출_금액": "주중_추정매출액", "주말_매출_금액": "주말_추정매출액"}
cols.update({f"시간대_{t.replace('_', '~')}_매출_금액": f"시간대_{t}_추정매출액" for t in TIME})
cols.update({f"{d}요일_매출_금액": f"{d}요일_추정매출액" for d in DAYS})
cols.update({"남성_매출_금액": "남성_추정매출액", "여성_매출_금액": "여성_추정매출액"})
cols.update({f"연령대_{a}_매출_금액": f"연령대_{a}_추정매출액" for a in ["10", "20", "30", "40", "50", "60_이상"]})
sa = sales.groupby(KEY, as_index=False)[list(cols)].sum().rename(columns=cols)
only_store = sorted(set(store["서비스_업종_코드_명"]) - set(sales["서비스_업종_코드_명"]))
only_sales = sorted(set(sales["서비스_업종_코드_명"]) - set(store["서비스_업종_코드_명"]))
log.append(f"추정매출: {sales.shape[0]:,}행 (업종 {n_ind_s}종) → 자치구 합산 {sa.shape[0]}행 "
           f"(원본 '당월_매출_금액'을 분기 총추정매출액으로 사용)")

# 4) 소득소비(지출) ────────────────────────────────────────
spend = key_cols(read(SPEND), c="행정동_코드", n="행정동_코드_명")
spend = spend.rename(columns={c: f"지출_{c.replace('_지출_총금액', '').replace('_총금액', '_총액')}"
                              for c in spend.columns if c not in KEY})
spend = spend.rename(columns={"지출_지출_총액": "지출_총금액"})
log.append(f"소득소비: {spend.shape[0]}행 (컬럼명 '행정동_코드' → 실제 자치구 코드라 자치구로 결합)")

# 5) 면적 ─────────────────────────────────────────────────
area = read(AREA).rename(columns={"자치구": "자치구명", "면적 (km²)": "면적_km2"})[["자치구명", "면적_km2"]]

# 6) 결합 ─────────────────────────────────────────────────
df = foot_out.merge(st, on=KEY, how="outer", validate="1:1", indicator="_m1")
assert (df["_m1"] == "both").all(), "유동인구-점포 키 불일치"
df = df.drop(columns="_m1").merge(sa, on=KEY, how="outer", validate="1:1", indicator="_m2")
assert (df["_m2"] == "both").all(), "점포-매출 키 불일치"
df = df.drop(columns="_m2").merge(spend, on=KEY, how="outer", validate="1:1", indicator="_m3")
assert (df["_m3"] == "both").all(), "매출-지출 키 불일치"
df = df.drop(columns="_m3").merge(area, on="자치구명", how="left", validate="m:1")
assert df["면적_km2"].notna().all(), "면적 결측"

front = ["년분기", "자치구코드", "자치구명", "면적_km2", "점포수", "개업률", "폐업률",
         "개업점포수", "폐업점포수", "프랜차이즈점포수", "총유동인구", "총추정매출액", "총추정매출건수"]
df = df[front + [c for c in df.columns if c not in front]].sort_values(["년분기", "자치구코드"])

# 7) 검증 ─────────────────────────────────────────────────
assert df.shape[0] == df["년분기"].nunique() * 25 and not df.duplicated(["년분기", "자치구코드"]).any()
assert df.isna().sum().sum() == 0
chk = {
    "유동인구 시간대합-총량 최대차(명)": (df[[f"시간대_{t}_유동인구" for t in TIME]].sum(axis=1) - df["총유동인구"]).abs().max(),
    "유동인구 요일합-총량 최대차(명)": (df[[f"{d}요일_유동인구" for d in DAYS]].sum(axis=1) - df["총유동인구"]).abs().max(),
    "매출 시간대합-총액 최대차(원)": (df[[f"시간대_{t}_추정매출액" for t in TIME]].sum(axis=1) - df["총추정매출액"]).abs().max(),
    "매출 요일합-총액 최대차(원)": (df[[f"{d}요일_추정매출액" for d in DAYS]].sum(axis=1) - df["총추정매출액"]).abs().max(),
}

df.to_csv(OUT / "data.csv", index=False, encoding="utf-8-sig")

# 컬럼 설명서
desc = []
for c in df.columns:
    src = ("키" if c in KEY else "면적표" if c == "면적_km2" else
           "점포" if c in st.columns else "추정매출" if c in sa.columns else
           "소득소비" if c in spend.columns else "유동인구")
    desc.append({"컬럼": c, "출처": src, "dtype": str(df[c].dtype)})
pd.DataFrame(desc).to_csv(OUT / "data_columns.csv", index=False, encoding="utf-8-sig")

print("\n".join(log))
print(f"\n결과 data.csv: {df.shape[0]}행 × {df.shape[1]}열, 기간 {df['년분기'].min()}~{df['년분기'].max()}")
for k, v in chk.items():
    print(f"  {k}: {v:,}")
print(f"\n점포에만 있는 업종 {len(only_store)}종 / 매출에만 있는 업종 {len(only_sales)}종")
print("  점포에만:", ", ".join(only_store[:12]), "…" if len(only_store) > 12 else "")
