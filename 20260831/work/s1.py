# 테스트
from pathlib import Path
import numpy as np
import pandas as pd
import csv

BASE = Path(__file__).parent
csv_path = BASE / "설비배치1.csv"

df = pd.read_csv(csv_path, encoding="utf-8-sig")
센서 = ["온도", "진동", "회전수", "압력"]

# 문제 1. 받은 파일을 열고 상태를 파악한다
# ----------------------------------------
# 설비배치1.csv 를 읽어 표 크기, 결측이 있는 열과 개수, 진동 열 첫 값의 자료형 이름,
# 판정 열의 값별 개수를 차례로 출력하세요. 한글이 깨지지 않게 읽습니다.


# print("실습 데이터 준비 완료:", csv_path.name)
# print(df.shape)
miss = df.isnull().sum()  # isnull이 결측치같은거 True로 반환 sum으로 합산
# print(miss[miss > 0].to_dict())  # 앞에를 키로 뒤에를 밸류
# print(df["진동"].dtypes)
# print(df["판정"].value_counts().to_dict())  # 특정 열의 값과 개수 가져오기(내림차순)

# 문제 2. 숫자로 저장되지 않은 열 고치기
# ----------------------------------------
# 진동 열을 숫자로 바꾸세요. 숫자로 못 바꾸는 값은 결측으로 만듭니다.
# 바꾼 뒤 진동 열의 결측 개수와 평균(소수 둘째 자리)을 출력하세요.

df["진동"] = pd.to_numeric(df["진동"], errors="coerce")
# print(df["진동"].isnull().sum())
# print(round(df["진동"].mean(), 2))

# 문제 3. 중복 행 제거
# ----------------------------------------
# 완전히 같은 행이 몇 개인지 출력하고, 지운 뒤 표 크기를 출력하세요.
# 인덱스는 0부터 다시 매깁니다.

# print("\n중복 행 개수:", df.duplicated().sum())  # → 중복 행 개수: 1
df = df.drop_duplicates()  # 결과는 다시 담기 (원본을 안 바꿈)
# print("중복 제거 후 행 열:", df.shape)  # → 중복 제거 후 행 수: 14

# 문제 4. 결측 채우기
# ----------------------------------------
# 온도는 온도 열 전체 평균으로, 압력은 압력 열 전체 중앙값으로,
# 진동은 진동 열 전체 평균으로 채우세요.
# 채운 뒤 센서 4열의 남은 결측 총 개수를 출력하고,
# 채우는 데 쓴 온도 평균과 압력 중앙값을 소수 둘째 자리로 한 줄에 출력하세요.


df["온도"] = df["온도"].fillna(df["온도"].mean())
df["압력"] = df["압력"].fillna(df["압력"].median())
df["진동"] = df["진동"].fillna(df["진동"].mean())
print(int(df["압력"].isnull().median()))
print(round(df["온도"].mean(), 1), df["압력"].median())


# 문제 5. 생산라인별 요약
# ----------------------------------------
# 생산라인별 센서 4종 평균(소수 둘째 자리)을 표로 출력하고,
# 라인별 검사 건수를 라인 이름 순으로 출력하세요.

print(df.groupby("생산라인")[["온도", "진동", "회전수", "압력"]].mean().round(2))
print(df["생산라인"].value_counts().sort_index().to_dict())

# 문제 6. z-점수로 온도 이상 찾기
# ----------------------------------------
# 온도 열 전체의 평균과 표준편차(ddof=0)를 소수 둘째 자리로 한 줄에 출력하세요.
# 이어서 z-점수 절댓값이 3을 넘는 개수와 2를 넘는 개수를 한 줄에 출력하세요.

print(df["온도"].mean().round(2), df["온도"].std(ddof=0).round(2))
z = (df["온도"] - df["온도"].mean()) / df["온도"].std(ddof=0)

print(int(z[np.abs(z) > 3].sum()), int(z[np.abs(z) > 2].sum()))

# 문제 7. IQR로 압력 이상 찾기
# ----------------------------------------
# 압력 열 전체의 아래·위 울타리(임계값 1.5)를 소수 둘째 자리로 한 줄에 출력하고,
# 울타리를 벗어난 개수를 출력하세요.
# 이어서 걸린 행이 어느 생산라인에서 나왔는지 개수를 출력하세요.

q1 = np.percentile(df["압력"], 25)
q3 = np.percentile(df["압력"], 75)
high = q3 + 1.5 * (q3 - q1)
low = q1 - 1.5 * (q3 - q1)
# print(round(low, 2), round(high, 2))
# print("IQR 이상값:", ((df["압력"] < low) | (df["압력"] > high)).sum())
mask = (df["압력"] < low) | (df["압력"] > high)
result1 = df.loc[mask, "생산라인"].value_counts().to_dict()
# print(result1)

# 문제 8. 이상으로 판정된 행 제거
# ----------------------------------------
# 문제 7에서 걸린 행을 표에서 제거하세요.
# 제거 전 라인별 행 수, 제거 후 라인별 행 수, 그리고 최종 표 크기를 차례로 출력하세요.
# 인덱스는 다시 매깁니다.
# print(df["생산라인"].value_counts().sort_index().to_dict())
df = df.drop(df[mask].index)
# print(df["생산라인"].value_counts().sort_index().to_dict())
# print(df.shape)

# 문제 9. 0~1로 스케일 맞추고 파일로 남기기
# ----------------------------------------
# 센서 4열을 표 전체의 최솟값·최댓값을 기준으로 Min-Max 정규화하세요.
# 열별 최솟값, 최댓값, 평균을 소수 셋째 자리로 차례로 출력하세요.
# 이어서 검사일시·생산라인 두 열 뒤에 정규화한 센서 4열(소수 넷째 자리)을 붙여
# 정규화_멘티.csv 로 저장하고, 다시 읽어 표 크기를 출력하세요.

df_normalized = round(
    (df[센서] - df[센서].min()) / (df[센서].max() - df[센서].min()), 2
)

a = df_normalized.min(axis=0)
b = df_normalized.max(axis=0)
c = round(df_normalized.mean(axis=0), 2)
print(a.to_dict())
print(b.to_dict())
print(c.to_dict())
# print(df.iloc[:, :2])
# .value_counts().to_dict()
s = df.iloc[:, :2]

y = np.concatenate([s, df_normalized], axis=1)
# print(y)
print(y.shape)

# 기존 헤더에 열 추가

with open("설비배치1.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(y)
    # 기존 헤더에 열 추가

with open("정규화_멘티.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)

    # 헤더 한 줄 추가
    writer.writerow(
        ["검사일시", "생산라인", "설비번호", "온도", "진동", "회전수", "압력", "판정"]
    )

    # y의 내용 저장
    writer.writerows(y)

# 문제 10. 라인 인코딩하고 저장하기
# ----------------------------------------
# 생산라인을 A라인 0, B라인 1, C라인 2 로 바꾼 라인코드 열을 만드세요.
# 검사일시, 생산라인, 라인코드, 온도, 진동, 회전수, 압력, 판정 순으로 열을 골라

# 생산라인을 숫자로 인코딩
df["라인코드"] = df["생산라인"].replace({"A라인": 0, "B라인": 1, "C라인": 2})

# 열 순서 지정
columns = ["검사일시", "생산라인", "라인코드", "온도", "진동", "회전수", "압력", "판정"]

df = df[columns]

# CSV 저장 (인덱스 없이, 한글 깨짐 방지)
df.to_csv("정제결과_멘티.csv", index=False, encoding="utf-8-sig")

# 저장한 파일 다시 읽기
check_df = pd.read_csv("정제결과_멘티.csv", encoding="utf-8-sig")

# 표 크기, 결측 총 개수, 중복 개수 출력
print(check_df.shape, check_df.isna().sum().sum(), check_df.duplicated().sum())

# 열 이름 출력
print(check_df.columns.tolist())
