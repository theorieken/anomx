"""Replace TRAINING_CODE with the adapted forecast.py contents before saving Work."""

TRAINING_CODE = """
raise ValueError("Insert the inspected dataset mapping and full training procedure.")
"""

work.log("Dispatching training to the selected compute target")
training_result = work.compute(TRAINING_CODE, inputs=inputs)
# Custom orchestration code is part of this saved procedure and every run snapshot.
metrics = training_result["metrics"]
work.log(
    "Forecast evaluation finished",
    test_rmse=metrics["test_rmse"],
    persistence_rmse=metrics["persistence_rmse"],
)
result = training_result
