import pytest
from threadpoolctl import threadpool_limits

from fraud_benchmark.data import generate_transactions
from fraud_benchmark.experiment import ExperimentConfig, run_experiment


@pytest.fixture(autouse=True)
def limited_threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def dataset():
    return generate_transactions(2000, 2026)


@pytest.fixture(scope="session")
def completed_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("benchmark") / "run"
    report = run_experiment(output, ExperimentConfig(rows=2000))
    return output, report
