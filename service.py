import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    datetime: int
    season: int
    holiday: int
    workingday: int
    weather: int
    temp: float
    atemp: float
    humidity: int
    windspeed: float


# 학습 코드에서 저장한 베스트 모델을 가져올 것 (bike_sharing:latest)
bento_model = bentoml.sklearn.get("bike_sharing:latest")
model_runner = bento_model.to_runner()

# "bike_sharing_regressor"라는 이름으로 서비스를 띄우기
svc = bentoml.Service("bike_sharing_regressor", runners=[model_runner])

@svc.api(
    # Features 클래스를 JSON으로 받아오고 Numpy NDArray를 반환하도록 데코레이터 작성
    input=JSON(pydantic_model=Features),
    output=NumpyNdarray(),
)
async def predict(input_data: Features) -> npt.NDArray:
    input_df = pd.DataFrame([input_data.dict()])
    log_pred = await model_runner.predict.async_run(input_df)
    return np.expm1(log_pred)
