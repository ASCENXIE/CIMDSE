"""Load experiment-local SRAM-CIM YAML configurations into AccelForge."""

from pathlib import Path

import accelforge as af


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "SRAM_config"
ACCELFORGE_CIM_DIR = (
    REPO_ROOT / "third_party" / "accelforge" / "examples" / "arches" / "compute_in_memory"
)
RUN_DIR = Path(__file__).resolve().parent


def load_sram_spec(config_name: str) -> af.Spec:
    """Load an experiment-local YAML architecture with standard CIM mapper settings."""

    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"SRAM-CIM configuration not found: {config_path}")

    variables = af.Variables.from_yaml(config_path, top_key="variables")
    arch = af.Arch.from_yaml(config_path, top_key="arch")
    workload = af.Workload.from_yaml(config_path, top_key="workload")
    spec = af.Spec(arch=arch, variables=variables, workload=workload)
    spec.mapper.explore_imperfect_spatial_loops = True
    spec.mapper.explore_loop_orders = False
    spec.mapper.metrics = af.Metrics.ENERGY
    spec.config.expression_custom_functions.extend(
        [
            str(ACCELFORGE_CIM_DIR / "_include_functions.py"),
            str(RUN_DIR / "sram_helper_functions.py"),
        ]
    )
    spec.config.component_models.append(str(RUN_DIR / "sram_component_models.py"))
    return spec
