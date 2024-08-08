import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "season",
    "holiday",
    "workingdy",
    "weather"
]


def extract_time(datetime: str) -> int:
    """
    시간 만 추출하여 정수 반환
    """
    split_datetime_info = datetime.split()[1].split(':')
    time_str = split_datetime_info[0]

    return int(time_str) if time_str.isnumeric() else 0


def time_extractor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """`extract_floor()` 함수를 `FunctionTransformer`에 사용하기 위한
    Wrapping function입니다.

    Args:
        df (pd.DataFrame): 데이터프레임
        col (str): `extract_time()`를 적용할 컬럼명
            `datetime`만 사용해야 함

    Returns:
        pd.DataFrame: 컬럼 처리 후 데이터
    """
    df[col] = df[col].apply(lambda x: extract_time(x))
    return df


# 전처리 파이프라인 작성
# 1. 방의 크기는 제곱근을 적용함 (FunctionTransformer 사용)
# 2. 층수는 실제 층수를 추출하되 숫자가 아닌 Basement 등은 0층으로 표기함
# 3. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용 (from category_encoders import TargetEncoder)
preprocess_pipeline = ColumnTransformer(
    transformers=[
        (
            "time_extractor",
            FunctionTransformer(time_extractor, kw_args={"col": "datetime"}),
            ["datetime"],
        ),
        ("target_encoder", TargetEncoder(cols=CAT_FEATURES), CAT_FEATURES),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
