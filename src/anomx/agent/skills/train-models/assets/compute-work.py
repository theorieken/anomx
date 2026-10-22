"""Save as main.py beside native-forecast.py copied as training.py.

Only standard-library coordination runs on the task worker. Dataset reads,
scientific imports, fitting, scoring and artifact creation run on the selected
compute target. Both files are frozen in the shared job workspace for each Run.
"""

# The Work runner injects work and inputs into this entry point.
result = work.compute("from training import run\nresult = run(inputs, work)", inputs=inputs)  # noqa: F821
work.log("Forecast evaluation finished", metrics=result["metrics"])  # noqa: F821
