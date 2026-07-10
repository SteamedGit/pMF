"""Main file for running the ImageNet experiments."""

import os

import jax

# Distributed init is only needed for multi-process (multi-node) runs,
# On a single local GPU we skip it entirely: there is no
# cluster manager for jax to auto-detect, and jax uses the local device(s) directly.
if os.environ.get("JAX_COORDINATOR_ADDRESS"):
    # Explicit launch: export these in your job script (one process per node).
    # For one-process-per-GPU, also pass local_device_ids=[<gpu_index>].
    jax.distributed.initialize(
        coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
        process_id=int(os.environ["JAX_PROCESS_ID"]),
    )
    print(f"JAX distributed initialised: {jax.process_index()}/{jax.process_count()}")
elif os.environ.get("SLURM_JOB_ID"):
    # SLURM auto-detects coordinator/num_processes/process_id from SLURM_* env vars.
    jax.distributed.initialize()
    print(
        f"JAX distributed initialised (SLURM): {jax.process_index()}/{jax.process_count()}"
    )
else:
    print("Single-process mode: skipping jax.distributed.initialize().")
from absl import app, flags
from ml_collections import config_flags

import train
from utils import logging_util
from utils.logging_util import log_for_0

logging_util.supress_checkpt_info()

import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("debug", False, "Debugging mode.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    log_for_0("JAX process: %d / %d", jax.process_index(), jax.process_count())
    log_for_0("JAX local devices: %r", jax.local_devices())
    log_for_0("FLAGS.config: \n{}".format(FLAGS.config))

    if FLAGS.config.eval_only:
        train.just_evaluate(FLAGS.config, FLAGS.workdir)
    else:
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
