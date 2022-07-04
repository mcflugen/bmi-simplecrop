import contextlib
import os
import pathlib
import shutil
import subprocess

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd

from bmipy import Bmi
from model_metadata.scripting import as_cwd


class SimpleCropError(Exception):
    pass


class SimpleCropNotFoundError(SimpleCropError):
    def __init__(self, msg):
        self._msg = msg
    def __str__(self):
        return self._msg

PLANT_INPUT = {
    "n_leaves_max": 12.0,
    "coef_1": 0.64,
    "coef_2": 0.1040,
    "plant_density": 5.0,
    "coef_3": 5.3,
    "leaf_appearance_rate_max": 0.1,
    "canopy_crop_growth_fraction": 0.85,
    "temperature_base": 10.0,
    "reproductive_stage_days": 300.0,
    "leaf_number": 2.0,
    "leaf_area_index": 0.013,
    "dry_matter_weight": 0.3,
    "root_dry_matter_weight": 0.045,
    "canopy_dry_matter_weight": 0.255,
    "removed_dry_matter_weight": 0.3,
    "f1": 0.028,
    "specific_leaf_area": 0.35,
}


def load_irrigation(filepath):
    return pd.read_fwf(
        filepath,
        widths=[2, 3, 10],
        names=["year", "day_of_year", "irrigation_rate"],
        header=None,
    )


def load_soil(filepath):
    return pd.read_fwf(
        filepath,
        colspecs="infer",
        names=[
            "wilting_point_water_content",
            "field_capacity_water_content",
            "saturation_water_content",
            "profile_depth",
            "daily_drainage_percentage",
            "runoff_curve_number",
            "soil_water_storage",
        ],
        header=None,
        nrows=1,
    )


def load_plant(filepath):
    return pd.read_fwf(
        filepath,
        colspecs="infer",
        names=[
            "n_leaves_max",
            "coef_1",
            "coef_2",
            "plant_density",
            "coef_3",
            "leaf_appearance_rate_max",
            "canopy_crop_growth_fraction",
            "temperature_base",
            "reproductive_stage_days",
            "leaf_number",
            "leaf_area_index",
            "dry_matter_weight",
            "root_dry_matter_weight",
            "canopy_dry_matter_weight",
            "removed_dry_matter_weight",
            "f1",
            "specific_leaf_area",
        ],
        header=None,
        nrows=1,
    )


def load_weather(filepath):
    return pd.read_fwf(
        filepath,
        widths=[2, 3, 6, 6, 6, 6, 20],
        names=[
            "year",
            "day_of_year",
            "solar_radiation",
            "temperature_max",
            "temperature_min",
            "rainfall",
            "active_radiation",
        ],
    )


def validate_config_file_path(filepath):
    filepath = pathlib.Path(filepath).absolute()
    if filepath.name != "simctrl.inp":
        raise ValueError(
            f"{filepath.name}: invalid name for config file, must be 'simctrl.inp'"
        )
    if filepath.parts[-2] != "data":
        raise ValueError(
            f"{filepath.parts[-2]}: invalid folder for config file, must be 'data'"
        )


class SimpleCrops:

    """Run multiple simplecrop simulations in parallel."""

    def __init__(self, filepaths):
        # filepath = pathlib.Path(filepath)
        # relative_to = filepath.parent.absolute()

        # with open(filepath, "r") as fp:
        #     filepaths = [
        #         relative_to / pathlib.Path(line.strip())
        #         for line in fp.readlines()
        #         if line.strip()
        #     ]
        self._crops = db.from_sequence(filepaths).map(
            lambda filepath: SimpleCrop(filepath)
        )
        self._n_crops = len(filepaths)

        self._output = dict()

    @classmethod
    def from_file(cls, filepath):
        filepath = pathlib.Path(filepath)
        relative_to = filepath.parent.absolute()

        with open(filepath, "r") as fp:
            filepaths = [
                relative_to / pathlib.Path(line.strip())
                for line in fp.readlines()
                if line.strip()
            ]
        return cls(filepaths)

    @property
    def n_crops(self):
        return self._n_crops

    def __len__(self):
        return self.n_crops

    def run(self):
        self._crops.map(lambda crop: crop.run()).compute()
        self._output["soil"] = self.load_soil()

    def load_soil(self):
        self._output["soil"] = SimpleCrops.read_soil_output(
            [pathlib.Path(crop.run_dir) / "output" / "soil.out" for crop in self._crops]
        )
        return self._output["soil"]

    @staticmethod
    def read_soil_output(filepath):
        return dd.read_fwf(
            filepath,
            names=[
                "day_of_year",
                "solar_radiation",
                "temperature_max",
                "temperature_min",
                "rain",
                "irrigation",
                "runoff",
                "infiltration",
                "drain",
                "evapotranspiration",
                "soil_evaporation",
                "plant_transpiration",
                "soil_water_content",
                "soil_water_content_concentration",
                "drought_stress",
                "excess_water_stress",
            ],
            header=None,
            skiprows=6,
            widths=[5] + [8] * 3 + [8] * 9 + [8] * 3,
            na_values="*" * 8,
            include_path_column=True,
            blocksize=None,
        )


class SimpleCrop:

    """Run a simplecrop simulation."""

    REQUIRED_FOLDERS = ["data", "output"]
    REQUIRED_FILES = [
        "data/irrig.inp",
        "data/plant.inp",
        "data/simctrl.inp",
        "data/soil.inp",
        "data/weather.inp",
    ]

    def __init__(self, run_dir):
        self._program = SimpleCrop.which()
        self._run_dir = SimpleCrop.validate_run_dir(run_dir)

        self._output = dict()

    @staticmethod
    def which():
        return shutil.which(os.environ.get("SIMPLECROP", "simplecrop"))

    @classmethod
    def from_config_file(cls, config_file):
        return cls(pathlib.Path(config_file).parent.parent)

    @property
    def run_dir(self):
        return str(self._run_dir)

    @property
    def simplecrop(self):
        return str(self._program)

    @staticmethod
    def validate_run_dir(run_dir):
        run_dir = pathlib.Path(run_dir)
        for name in SimpleCrop.REQUIRED_FOLDERS:
            if not (run_dir / name).is_dir():
                raise ValueError(f"{run_dir / name}: missing required folder")
        for name in SimpleCrop.REQUIRED_FILES:
            if not (run_dir / name).is_file():
                raise ValueError(f"{run_dir / name}: missing required input file")
        return pathlib.Path(run_dir).resolve()

    def run(self):
        with as_cwd(self._run_dir):
            subprocess.run(self._program, capture_output=True, check=True)
        self._output["soil"] = self.load_soil()

    @property
    def soil_water_content(self):
        return np.asarray(self._output["soil"]["soil_water_content"])

    def load_plant(self):
        """
        Results of plant growth simulation::

                        Accum
               Number    Temp                                    Leaf
          Day      of  during   Plant  Canopy    Root   Fruit    Area
           of    Leaf  Reprod  Weight  Weight  Weight  weight   Index
         Year   Nodes    (oC)  (g/m2)  (g/m2)  (g/m2)  (g/m2) (m2/m2)
         ----  ------  ------  ------  ------  ------  ------  ------
          121    2.00    0.00    0.30    0.25    0.05    0.00    0.01
          123    2.20    0.00    0.65    0.55    0.10    0.00    0.02

        """
        return pd.read_fwf(
            pathlib.Path(self.run_dir) / "output" / "plant.out",
            colspecs="infer",
            names=[
                "day_of_year",
                "n_leaf_nodes",
                "accumulated_temperature",
                "plant_weight",
                "canopy_weight",
                "root_weight",
                "fruit_weight",
                "leaf_area_index",
            ],
            header=None,
            skiprows=9,
        )

    @staticmethod
    def read_soil_output(filepath):
        """
        Results of soil water balance simulation::

                                                                                                                 Soil
                                                                                 Pot.  Actual  Actual    Soil   Water          Excess
          Day   Solar     Max     Min                                          Evapo-    Soil   Plant   Water Content Drought   Water
           of    Rad.    Temp    Temp    Rain   Irrig  Runoff   Infil   Drain   Trans   Evap.  Trans. content   (mm3/  Stress  Stress
         Year (MJ/m2)    (oC)    (oC)    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)    mm3)  Factor  Factor
            0     0.0********     0.0    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00  246.50   1.700   1.000   1.000
            3    12.1    14.4     1.1    0.00    0.00    0.00    0.00    1.86    2.25    2.23    0.02  260.97   1.800   1.000   1.000
            6    12.4    21.1     5.0    0.00    0.00    0.00    0.00    2.25    2.64    2.62    0.02  264.09   1.821   1.000   1.000
            9     6.1    20.0     8.3    0.00    0.00    0.00    0.00    0.94    1.32    1.31    0.01  253.64   1.749   1.000   1.000

        """
        return pd.read_fwf(
            filepath,
            names=[
                "day_of_year",
                "solar_radiation",
                "temperature_max",
                "temperature_min",
                "rain",
                "irrigation",
                "runoff",
                "infiltration",
                "drain",
                "evapotranspiration",
                "soil_evaporation",
                "plant_transpiration",
                "soil_water_content",
                "soil_water_content_concentration",
                "drought_stress",
                "excess_water_stress",
            ],
            header=None,
            skiprows=6,
            widths=[5] + [8] * 3 + [8] * 9 + [8] * 3,
            na_values="*" * 8,
        )

    def load_soil(self):
        return SimpleCrop.read_soil_output(pathlib.Path(self.run_dir) / "output" / "soil.out")

    def __repr__(self):
        return f"SimpleCrop({self.run_dir!r})"
