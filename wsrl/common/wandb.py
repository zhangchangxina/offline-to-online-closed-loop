import datetime
import os
import random
import string
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import numpy as np
import wandb

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


def _filter_nan_values(data: dict) -> dict:
    """
    Recursively filter out NaN and infinite values from a dictionary.
    This prevents wandb from crashing when trying to create histograms from NaN values.
    """
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively filter nested dictionaries
            filtered_value = _filter_nan_values(value)
            if filtered_value:  # Only include non-empty dictionaries
                filtered_data[key] = filtered_value
        else:
            # Check if value is a number and if it's finite
            try:
                # Handle JAX arrays if available
                if HAS_JAX and hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    # This is likely a JAX array
                    if jnp.isfinite(value).all():
                        filtered_data[key] = value
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle numpy arrays/lists
                    value_array = np.asarray(value)
                    if np.isfinite(value_array).all():
                        filtered_data[key] = value
                else:
                    # Handle scalar values
                    if np.isfinite(value):
                        filtered_data[key] = value
            except (TypeError, ValueError, AttributeError):
                # If we can't convert to numeric, keep the original value
                # (e.g., strings, booleans, etc.)
                filtered_data[key] = value
    return filtered_data


def generate_random_string(length=6):
    # Define the character set for the random string
    characters = string.digits  # Use digits 0-9

    # Generate the random string by sampling from the character set
    random_string = "".join(random.choices(characters, k=length))

    return "rnd" + random_string


# Generate a 6-digit random string
random_string = generate_random_string()
print(random_string)


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "wsrl"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        random_str_in_identifier=False,
        wandb_output_dir=None,
        disable_online_logging=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            if random_str_in_identifier:
                self.config.unique_identifier += "_" + generate_random_string()

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if disable_online_logging:
            mode = "disabled"
        else:
            mode = os.environ.get("WANDB_MODE", "online")

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        # Filter out NaN and infinite values before logging
        filtered_data = _filter_nan_values(data)
        
        # Log a warning if any values were filtered out
        original_keys = set(_recursive_flatten_dict(data)[0])
        filtered_keys = set(_recursive_flatten_dict(filtered_data)[0])
        filtered_out_keys = original_keys - filtered_keys
        if filtered_out_keys:
            print(f"Warning: Filtered out {len(filtered_out_keys)} metrics with NaN/inf values: {list(filtered_out_keys)}")
        
        data_flat = _recursive_flatten_dict(filtered_data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)
