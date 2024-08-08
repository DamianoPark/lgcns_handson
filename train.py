# sourcery skip: raise-specific-error
import os
import sys
import warnings
from distutils.dir_util import copy_tree

import bentoml
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.common.constants import ARTIFACT_PATH, DATA_PATH, LOG_FILEPATH
from src.common.logger import (
    handle_exception,
    log_feature_importance,
    set_logger,
)
from src.common.metrics import rmse_cv_score
from src.common.utils import get_param_set
from src.preprocess import preprocess_pipeline

# 로그 들어갈 위치
# TODO: 로그를 정해진 로그 경로에 logs.log로 저장하도록 설정

logger = set_logger(log_path=os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_PATH, "bike_sharing_train.csv"))
    logger.debug("Load data...")

    _X = train_df.drop(["count"], axis=1)
    y = np.log1p(train_df["count"])

    # X=_X, y=y로 전처리 파이프라인을 적용해 X에 저장
    X = preprocess_pipeline.fit_transform(X=_X, y=y)

    # Data storage - 피처 데이터 저장
    if not os.path.exists(os.path.join(DATA_PATH, "storage")):
        os.makedirs(os.path.join(DATA_PATH, "storage"))
    X.assign(rent=y).to_csv(
        # DATA_PATH 밑에 storage 폴더 밑에 피처 데이터를 저장
        os.path.join(DATA_PATH, "storage", "bike_sharing_train_features.csv"),
        index=False,
    )
    logger.debug("Save the feature data...")

    params_candidates = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5 ],
        "max_features": [1.0, 0.9, 0.8],
    }

    param_set = get_param_set(params=params_candidates)

    # Set experiment name for mlflow
    experiment_name = "new_experiment_with_log"
    mlflow.set_experiment(experiment_name=experiment_name)
    logger.info("Set an mlflow experiment...")
    mlflow.set_tracking_uri("./mlruns")

    for i, params in enumerate(param_set):
        run_name = f"Run {i}"
        logger.info(f"{run_name}: {params}")
        with mlflow.start_run(run_name=f"Run {i}"):
            regr = GradientBoostingRegressor(**params)
            # 전처리 이후 모델 순서로 파이프라인 작성
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocess_pipeline),
                    ("gradient_booster", regr),
                ]
            )
            pipeline.fit(_X, y)

            # get evaluations scores
            score_cv = rmse_cv_score(regr, X, y)
            logger.info(f"CV RMSE score for Run {i}: {score_cv.mean():.4f}")

            name = regr.__class__.__name__
            mlflow.set_tag("estimator_name", name)

            # 로깅 정보 : 파라미터 정보
            mlflow.log_params({key: regr.get_params()[key] for key in params})

            # 로깅 정보: 평가 메트릭
            mlflow.log_metrics(
                {
                    "RMSE_CV": score_cv.mean(),
                }
            )

            # 로깅 정보 : 학습 loss
            for s in regr.train_score_:
                mlflow.log_metric("Train Loss", s)

            # 모델 아티팩트 저장
            mlflow.sklearn.log_model(
                pipeline,
                "model",
            )

            # log charts
            mlflow.log_artifact(ARTIFACT_PATH)

            # generate a chart for feature importance
            log_feature_importance(train=X, model=regr)

    # Find the best regr
    best_run_df = mlflow.search_runs(
        order_by=["metrics.RMSE_CV ASC"], max_results=1
    )

    if len(best_run_df.index) == 0:
        raise Exception(f"Found no runs for experiment '{experiment_name}'")

    best_run = mlflow.get_run(best_run_df.at[0, "run_id"])
    best_params = best_run.data.params
    logger.info(f"Best hyper-parameter: {best_params}")

    best_model_uri = f"{best_run.info.artifact_uri}/model"

    # 베스트 모델을 아티팩트 폴더에 복사
    copy_tree(best_model_uri.replace("file://", ""), ARTIFACT_PATH)

    # BentoML에 모델 저장
    bentoml.sklearn.save_model(
        name="bike_sharing",
        model=mlflow.sklearn.load_model(best_model_uri),
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
        metadata=best_params,
    )
