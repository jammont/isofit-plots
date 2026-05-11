"""
ISOFIT Output Parser
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime as dtt
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from spectral.io import envi as _envi
from xarray.backends import BackendEntrypoint

from isofit.radiative_transfer import luts


class Loaders:
    """
    Collection of loader functions for common products of ISOFIT
    """

    @classmethod
    def text(cls, file):
        """
        Loads the lines of a text file

        Parameters
        ----------
        file : str
            File to load

        Returns
        -------
        list[str]
            file.readlines()
        """
        with open(file) as f:
            return f.readlines()

    @classmethod
    def json(cls, file):
        """
        Loads a JSON file

        Parameters
        ----------
        file : pathlib.Path
            Path to file to load

        Returns
        -------
        dict
            Loaded JSON dict
        """
        with open(file, "rb") as f:
            return json.load(f)

    @classmethod
    def lut(cls, file):
        return luts.load(file)

    @classmethod
    def envi(cls, file):
        """
        Loads an ENVI file

        Parameters
        ----------
        file : pathlib.Path
            Path to file to load

        Returns
        -------
        xr.Dataset | xr.DataArray
            Loaded xarray object from the ENVI. If the Dataset is only one variable,
            returns the DataArray of that variable instead
        """
        file = Path(file)
        if file.suffix:
            file = file.with_suffix("")

        ds = xr.open_dataset(file, engine="envi")

        # Return the DataArray if it is a dataset of one variable
        # Such as cases of `band_data`
        if len(ds) == 1:
            return ds[list(ds)[0]]
        return ds

    @classmethod
    def csv(cls, file, *args, **kwargs):
        """
        Loads a CSV file using Pandas

        Parameters
        ----------
        file : pathlib.Path
            Path to file to load
        *args : list, default=[]
            Additional arguments to pass to pd.read_csv
        **kargs : dict, default={}
            Additional key-word arguments to pass to pd.read_csv

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of the CSV
        """
        return pd.read_csv(file, *args, **kwargs)


class EnviBackendEntrypoint(BackendEntrypoint):
    """
    Uses spectral.io.envi to load ISOFIT output rasters

    Accessible via engine="envi" after installing ISOFIT

    Specifically structured to emulate the return of rasterio's engine for xarray to
    ease switching between them
    """

    description = "Uses spectral.io.envi to load"
    url = None

    def open_dataset(self, filename_or_obj, *, drop_variables=None):
        """
        Parameters
        ----------
        filename_or_obj : str
            String path to an ISOFIT output product
        drop_variables : any, default=None
            Unused required parameter
        """
        filename_or_obj = Path(filename_or_obj)

        envi = _envi.open(filename_or_obj.with_suffix(".hdr"))
        data = envi.open_memmap()
        meta = envi.metadata.copy()

        dims = ("y", "x", "band")
        coords = {
            "band": range(1, data.shape[-1] + 1)
        }

        if "wavelength" in meta:
            wl = np.array(meta.pop("wavelength")).astype(float)
            fwhm = np.array(meta.pop("fwhm")).astype(float)

            # If the lengths match, tie these to the band dim
            if len(wl) == data.shape[-1]:
                coords["wavelength"] = ("band", wl)
                coords["fwhm"] = ("band", fwhm)
            else:
                meta["wavelength"] = wl
                meta["fwhm"] = fwhm

        ds = xr.Dataset({"band_data": (dims, data)}, coords=coords, attrs=meta)
        ds = ds.transpose("band", "y", "x")

        ds["band_data"].attrs = meta

        return ds


@dataclass(frozen=True)
class FileInfo:
    name: str
    info: Any


class FileFinder:
    """
    Utility class to find files under a directory using various matching strategies
    This must be subclassed and define the following:
        function _load       - The loader for a passed file
        attribute extensions - A list of extensions to retrieve
    """
    path = None
    cache = None
    patterns = {}
    _compiled = []
    extensions = []

    def __init__(self, path=None, cache=True, extensions=[], patterns={}):
        """
        Parameters
        ----------
        path : str, default=None
            Path to directory to operate on
        cache : bool, default=True
            Enable caching objects in the .load function
        extensions : list, default=[]
            File extensions to retrieve when searching
        """
        if path is not None:
            self.path = Path(path)

        if extensions:
            self.extensions = extensions

        if not self.extensions:
            raise AttributeError("One or more extensions must be defined")

        if patterns:
            self.patterns = patterns

        if self.patterns:
            self._compiled = [(re.compile(p), desc) for p, desc in self.patterns.items()]

        if cache:
            self.cache = {}

        self.log = logging.getLogger(str(self))

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self.path}]>"

    def info(self, file):
        """
        Retrieves known information for a file if its name matches one of the set
        patterns

        Parameters
        ----------
        file : str
            File name to compare against the patterns dict keys

        Returns
        -------
        any
            Returns the value if a regex key in the patterns dict matches the file name
        """
        for pattern, desc in self._compiled:
            if pattern.search(file):
                return desc

    @cached_property
    def files(self):
        """
        Passthrough attribute to calling getFlat()
        """
        return self.getFlat()

    def extMatches(self, file):
        """
        Checks if a given file's extension matches in the list of extensions
        Special extension cases include:
            "*" = Match any file
            ""  = Only match files with no extension

        Parameters
        ----------
        file : pathlib.Path
            File path to check

        Returns
        -------
        bool
            True if it matches one of the extensions, False otherwise
        """
        try:
            if file.is_dir():
                return False

            if "*" in self.extensions:
                return True

            return file.suffix in self.extensions
        except:
            self.log.exception(f"Failed to evaluate extension match: {file}")

    def getTree(self, info=False, *, path=None, tree=None):
        """
        Recursively finds the files under a directory as a dict tree

        Parameters
        ----------
        info : bool, default=False
            Return the found files as objects with their respective info
        path : pathlib.Path, default=None
            Directory to search, defaults to self.path
        tree : dict, default=None
            Tree structure of discovered files

        Returns
        -------
        tree : dict
            Tree structure of discovered files. The keys are the directory names and
            the list values are the found files
        """
        if not (path := (path or self.path)):
            self.log.error("No path provided")
            return []

        path = Path(path)
        tree = tree if tree is not None else []
        try:
            with os.scandir(path) as scan:
                for item in scan:
                    name = item.name
                    if info:
                        name = FileInfo(name, self.info(name))

                    if item.is_dir():
                        subtree = []
                        tree.append({name: subtree})
                        self.getTree(info=info, path=item.path, tree=subtree)

                    elif self.extMatches(Path(item.path)):
                        tree.append(name)

        except Exception as e:
            self.log.exception(f"Error reading path {path}")

        return tree

    def getFlat(self, path=None):
        """
        Finds all the files under a directory as a flat list with the base path removed

        Parameters
        ----------
        path : pathlib.Path or None
            Root path to search, defaults to self.path

        Returns
        -------
        files : list[str]
            Flat list of matching file paths relative to base
        """
        if not (path := (path or self.path)):
            self.log.error("No path provided")
            return []

        base = Path(path)
        base_len = len(str(base)) + 1  # precompute for slicing
        files = []
        try:
            for root, _, filenames in os.walk(base):
                for name in filenames:
                    path = Path(root) / name
                    if self.extMatches(path):
                        files.append(str(path)[base_len:])
        except:
            self.log.exception(f"Error scanning directory tree: {base}")

        return sorted(files)

    def ifin(self, name, all=False, exc=[]):
        """
        Simple if name in filename match

        Parameters
        ----------
        name : str
            String to check in the filename
        all : bool, default=False
            Return all files matched instead of the first instance
        exc : str | list[str], default=[]
            A string or list of strings to use to exclude files. If a file contains
            one of the strings in its name, it will not be selected

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        if isinstance(exc, str):
            exc = [exc]

        found = []
        for file in self.getFlat():
            if name in file:
                if not any(ex in file for ex in exc):
                    found.append(file)

        if not all and len(found) > 1:
            self.log.warning(
                "%d files matched pattern '%s'. Returning first match.", len(found), regex
            )

        return found if all else (found[0] if found else None)

    def match(self, regex, all=False, exc=[]):
        """
        Find files using a regex search

        Parameters
        ----------
        regex : str
            Regex pattern to search for
        all : bool, default=False
            Return all matches instead of first match
        exc : str or list of str
            Strings to exclude from results

        Returns
        -------
        str or list or None
            File path(s) matching the pattern
        """
        if isinstance(exc, str):
            exc = [exc]

        try:
            pattern = re.compile(regex)
        except re.error:
            self.log.exception("Invalid regex pattern: %s", regex)
            raise

        found = []
        for file in self.getFlat():
            if pattern.search(file):
                if not any(ex in file for ex in exc):
                    found.append(file)

        if not all and len(found) > 1:
            self.log.warning(
                "%d files matched pattern '%s'. Returning first match.", len(found), regex
            )

        return found if all else (found[0] if found else None)

    def find(self, name, *args, **kwargs):
        """
        Smart search for a file based on partial path structure

        Parameters
        ----------
        name : str
            Path-like name to construct fuzzy regex search
        *args, **kwargs : forwarded to match()

        Returns
        -------
        str or list or None
            Matched file(s)
        """
        # Escape user input to avoid accidental regex issues
        regex_parts = [f".*{re.escape(part)}.*" for part in name.split("/")]
        regex = "/".join(regex_parts)
        return self.match(regex, *args, **kwargs)

    def load(self, *, path=None, ifin=None, find=None, match=None, **kwargs):
        """
        Loads a file based on one of several matching strategies
        Only the first provided method is used

        Parameters
        ----------
        path : str
            Direct path to a file (absolute or relative to self.path)
        ifin : str
            Substring match against available files
        find : str
            Smart path-style fuzzy match
        match : str
            Regex match

        Returns
        -------
        any
            Loaded object from subclass's _load method
        """
        file = None
        if path is not None:
            file = path
        elif ifin is not None:
            file = self.ifin(ifin)
        elif find is not None:
            file = self.find(find)
        elif match is not None:
            file = self.match(match)
        else:
            raise AttributeError("One of the key-word arguments must be set")

        if file is None:
            raise FileNotFoundError("Cannot find file to load")

        if not (p := Path(file)).exists():
            p = self.path / file
            if not p.exists():
                raise FileNotFoundError("Cannot find file to load, attempted: %s", file)

        if self.cache is not None:
            if p not in self.cache:
                self.log.debug("Loading file: %s", p)
                data = self._load(p)
                if data is not None:
                    self.cache[p] = data

            self.log.debug("Returning from cache: %s", p)
            return self.cache.get(p)

        self.log.debug("Returning from load: %s", p)
        return self._load(p, **kwargs)

    def _load(self, file, **kwargs):
        raise NotImplementedError("Subclass must define this function")


class Config(FileFinder):
    extensions = [".json"]
    patterns = {
        # Presolve
        r"(.*_h2o.json)": "Presolve configuration produced by apply_oe",
        r"(.*_h2o.json.tmpl)": "Presolve configuration template for developer purposes",
        r"(.*_h2o_tpl.json)": "MODTRAN template configuration for the presolve run",
        # Full
        r"(.*_isofit.json)": "ISOFIT main configuration",
        r"(.*_isofit.json.tmpl)": "ISOFIT main configuration template for developer purposes",
        r"(.*_modtran_tpl.json)": "MODTRAN template configuration for ISOFIT",
    }

    _load = Loaders.json


class Data(FileFinder):
    extensions = [".mat", ".txt"]
    patterns = {
        r"(channelized_uncertainty.txt)": None,
        r"(model_discrepancy.mat)": None,
        r"(surface.mat)": None,
        r"(wavelengths.txt)": None,
    }

    def _load(self, file, **kwargs):
        if file.suffix == ".csv":
            return Loaders.csv(file, **kwargs)


class LUT(FileFinder):
    extensions = [".nc"]
    patterns = {
        r"(6S.lut.nc)": "LUT produced by the SixS radiative transfer model for sRTMnet",
        r"(lut.nc)": "Look-Up-Table for the radiative transfer model",
        r"(sRTMnet.predicts.nc)": "Output predicts of sRTMnet",
    }

    _load = Loaders.lut

    lut_regex = r"(\w+)-(\d*\.?\d+)_?"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parseLutFiles(self, ext):
        """
        Parses LUT_*.{ext} file names for the LUT grid

        Parameters
        ----------
        ext : str
            File extension to retrieve:
                sixs    = inp
                modtran = json

        Returns
        -------
        data : dict
            Quantities to a set of their LUT values parsed from the file names
        """
        data = {}
        for file in self.path.glob(f"*.{ext}"):
            matches = re.findall(self.lut_regex, file.stem[4:])  # [4:] skips LUT_
            for name, value in matches:
                quant = data.setdefault(name, set())
                quant.add(float(value))

        return data

    @cached_property
    def sixs(self):
        return self.parseLutFiles("inp")

    @cached_property
    def modtran(self):
        return self.parseLutFiles("json")


class Input(FileFinder):
    extensions = [""]

    _load = Loaders.envi

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        files = self.getFlat()

        for file in files:
            if not (self.path / file).with_suffix(".hdr").exists():
                raise FileNotFoundError(f"Missing .hdr file for {file}")


class Output(FileFinder):
    extensions = [""]
    patterns = {
        # Presolve
        r"(.*_subs_atm)": None,
        r"(.*_subs_h2o)": None,
        r"(.*_subs_rfl)": None,
        r"(.*_subs_state)": None,
        r"(.*_subs_uncert)": None,
        # Full
        r"(.*_atm_interp)": None,
        r"(.*_rfl)": "Reflectance",
        r"(.*_lbl)": None,
        r"(.*_uncert)": None,
    }
    _load = Loaders.envi

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        files = self.getFlat()

        self.name = None
        for file in files:
            if "subs" in file:
                self.h2o = True
            elif file.endswith("_rfl"):
                self.name = file[:-4]

        if self.name is None:
            self.log.warning(
                "Could not find the full reflectance product and therefore could not parse the name. This may have downstream effects"
            )

        self.products = {file.replace(f"{self.name}_", ""): file for file in files}

        for file in files:
            if not (self.path / file).with_suffix(".hdr").exists():
                raise FileNotFoundError(f"Missing .hdr file for {file}")

    def __getattr__(self, key):
        if key in self.products:
            return self.products[key]
        return super().__getattr__(key)

    def rgb(self, r=60, g=40, b=30):
        """
        Returns the RGB data of the RFL product

        Parameters
        ----------
        r : int, default=60
            Red band
        g : int, default=40
            Green band
        b : int, default=30
            Blue band

        Returns
        -------
        xr.DataArray
        """
        if not (file := self.find("rfl", exc="subs")):
            self.log.error(
                "Could not find the full reflectance product, does it exist?"
            )
            return

        data = self.load(path=file)

        # Retrieve the RGB subset
        rgb = data.sel(band=[r, g, b]).transpose("y", "x", "band")
        rgb /= rgb.max(["x", "y"])  # Brightens image

        # Convert to pixel coords for easier plotting
        rgb["x"] = range(rgb.x.size)
        rgb["y"] = range(rgb.y.size)

        return rgb


class Container:
    """
    Subclasses must contain the attributes:

    data : list
        Must be in the form of [(label[str], data[any])]
    dataclass : dict
        Must be in the form of {label[str]: dataclass}

    The dataclass must have these attributes:
        "enabled" : bool
        "lines" : list

    The dataclass must have the bool attribute 'enabled' to check when iterating over
    the data

    Additionally, subclasses may define the 'yields' attribute to control which
    variable is yielded during iter:
        "item"  = Yield the (label, data) pair [default]
        "label" = Yields the label string only
        "data"  = Yields only the data
    By default, "item" is yielded
    """
    yields = "item"

    def __iter__(self):
        """
        Iterates over the "data" attribute, yielding if that dataclass is enabled
        """
        for item in self.data:
            label, data = item
            if self.dataclass[label].enabled:
                yield locals()[self.yields]

    def __getattr__(self, label):
        return self[label]

    def __getitem__(self, label):
        return self.dataclass.get(label)

    def toggle(self, *labels, state):
        """
        Toggles dataclasses' enabled state

        Parameters
        ----------
        *labels : str
            Labels of the dataclasses to toggle
        state : bool
            Enabled state
        """
        for label in labels:
            if label in self.dataclass:
                self.dataclass[label].enabled = state
            else:
                self.log.error(f"Label not found: {label!r}")

    def enable(self, *args, **kwargs):
        """
        Toggles dataclasses' enabled state to True
        """
        self.toggle(*args, **kwargs, state=True)

    def disable(self, *args, **kwargs):
        """
        Toggles dataclasses' enabled state to False
        """
        self.toggle(*args, **kwargs, state=False)

    def reset(self):
        """
        Resets the "lines" attribute of each dataclass to an empty list
        """
        for label, dataclass in self.dataclass.items():
            if hasattr(dataclass, "lines"):
                dataclass.lines = []


@dataclass
class Marker:
    enabled: bool
    regex: Pattern
    format: str = None
    lines: List[dict] = field(default_factory=list)


class Markers(Container):
    def __init__(self):
        self.data = []
        self.dataclass = {
            "Presolve Start": Marker(
                enabled = True,
                regex = re.compile(r"Run ISOFIT initial guess"),
            ),
            "Full Solution Start": Marker(
                enabled = True,
                regex = re.compile(r"Running ISOFIT with full LUT"),
            ),
            "Inversions Start": Marker(
                enabled = True,
                regex = re.compile(r"Beginning \d+ inversions"),
            ),
            "Inversions End": Marker(
                enabled = False,
                regex = re.compile(r"Analytical line inversions complete"),
            ),
            "Analytical Start": Marker(
                enabled = True,
                regex = re.compile(r"Analytical line inference"),
            ),
            "Analytical End": Marker(
                enabled = False,
                regex = re.compile(r"Analytical line inversions complete .*"),
            ),
            "LUT Creation Start": Marker(
                enabled = True,
                regex = re.compile(r"Initializing LUT file"),
            ),
            "Loading LUT": Marker(
                enabled = True,
                regex = re.compile(r"Loading LUT into memory"),
            ),
            "LUT Loaded": Marker(
                enabled = True,
                regex = re.compile(r"LUTs fully loaded"),
            ),
            "Interpolators Built": Marker(
                enabled = True,
                regex = re.compile(r"Interpolators built"),
            ),
            "Sims Start": Marker(
                enabled = True,
                regex = re.compile(r"Executing parallel simulations"),
            ),
            "Sims End": Marker(
                enabled = True,
                regex = re.compile(r"100.00% simulations complete"),
            ),
            "Flushing": Marker(
                enabled = True,
                regex = re.compile(r"Flushing"),
            ),
            "Predicting sRTMnet": Marker(
                enabled = True,
                regex = re.compile(r"Loading and predicting with emulator"),
            ),
            "Resampling Parallel": Marker(
                enabled = True,
                regex = re.compile(r"Executing resamples in parallel"),
            ),
            "Resampling Finished": Marker(
                enabled = True,
                regex = re.compile(r"Resampling finished"),
            ),
            "Resampling": Marker(
                enabled = True,
                format = "Resampling {name}",
                regex = re.compile(r"Resampling (?P<name>.*)"),
            ),
        }

        self.log = logging.getLogger(self.__class__.__name__)

    def check(self, line):
        """
        Checks if the line's message matches one of the markers of interest and, if so,
        appends it to that marker's match list
        """
        for label, marker in self.dataclass.items():
            if (match := marker.regex.match(line["message"])):
                # data = match.groupdict()
                # if data and marker.format:
                #     label = marker.format.format(**data)

                marker.lines.append(line)
                self.data.append((label, line))

                # No need to check the other markers
                break


@dataclass
class Level:
    enabled: bool
    lines: List[dict] = field(default_factory=list)


class Levels(Container):
    yields = "data"

    def __init__(self):
        self.data = []
        self.dataclass = {
            "DEBUG": Level(enabled=True),
            "INFO": Level(enabled=True),
            "WARNING": Level(enabled=True),
            "ERROR": Level(enabled=True),
        }
        self.formats = {
            "timestamps": True,
            "extra padding": True, # Adds +1 to the level padding in build()
        }

        self.log = logging.getLogger(self.__class__.__name__)

    def toggle(self, *labels, state):
        """
        Overrides the inherited toggle function to check for "formats" keys first then
        call super()

        Parameters
        ----------
        labels : str | list[str]
            Labels of the dataclasses or formats to toggle
        state : bool
            Enabled state
        """
        labels = set(labels)
        formats = labels - set(self.dataclass)
        for label in formats:
            if label in self.formats:
                self.formats[label] = state

        super().toggle(*(labels - formats), state=state)

    def add(self, line):
        """
        Adds a line line dict to the appropriate dataclass and data attribute

        Parameters
        ----------
        line : dict
            Log line to add
        """
        level = line["level"]
        self.dataclass[level].lines.append(line)
        self.data.append((level, line))

    def build(self):
        """
        Builds the filtered contents dict into a list of tuples to be used for writing.
        Timestamps can be disabled by one of:

            self.formats["timestamps"] = False
            self.disable("timestamps")

        Returns
        -------
        lines : list[tuple[str, str, str]]
            Returns a list of 3-pair tuples of strings in the form:
                (timestamp, padded level, log message)
            Timestamp will be an empty string if it is not enabled
            The level is right-padded with whitespace to the length of the longest log
            level (eg. "warning", "debug  ")
        """
        # Get the amount of padding needed to make level labels be equal
        levels = [label
            for label, level in self.dataclass.items()
            if level.enabled and level.lines
        ]

        if not levels:
            self.log.warning("No lines were retrieved. This may be caused either by too strict filters or the log being empty")
            return []

        padding = len(max(levels)) + self.formats["extra padding"]

        lines = []
        for line in self:
            level = line["level"].ljust(padding)

            ts = ""
            if self.formats["timestamps"]:
                ts = line["timestamp"] + self.formats["extra padding"]

            lines.append([ts, level, line["message"]])

        return lines


class Logs(FileFinder):
    extensions = [".log"]
    patterns = {r"(.*)": "ISOFIT log file"}

    # Regex to detect the two types of ISOFIT log messages:
    # - "[level]:[timestamp] || [source] | [message]"
    # - "[level]:[timestamp] ||| [message]"
    logline = re.compile(
        r"(?P<level>[^:]+):"      # Match the log level (e.g., INFO), up to the colon
        r"(?P<timestamp>[^\s|]+)" # Match the timestamp, up to the first pipe, excluding the space
        r"\s*\|{2,3}\s*"          # Match either || or ||| with optional surrounding spaces
        r"(?:(?P<source>[^\s|]+)" # Optionally match the source, excluding the space
        r"\s*\|\s*)?"             # Followed by a single pipe and optional spaces
        r"(?P<message>.+)"        # Match the rest of the line as the message
    )

    # Base datetime to use for converting the relative timedeltas to a datetime
    base_dt = dtt.fromisocalendar(1, 1, 1)

    def __init__(self, file):
        self.file = file

        self.levels = Levels()
        self.markers = Markers()

        self.t0 = None

    def _load(self, file):
        """
        Streams lines from a growing text file, yielding each new line as soon as it is
        written
        """
        with open(file, "r") as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    yield None

    def parse(self, line, content=[]):
        """
        Parses an ISOFIT log file into a dictionary of content that can be used to
        filter and reconstruct lines into different formats
        """
        line = line.strip()

        if match := self.logline.match(line):
            parsed = match.groupdict()
            content.append(parsed)

            parsed["datetime"] = dtt.strptime(parsed["timestamp"], '%Y-%m-%d,%H:%M:%S')

            if not self.t0:
                self.t0 = parsed["datetime"]

            parsed["timedelta"] = parsed["datetime"] - self.t0
            parsed["relative_datetime"] = self.base_dt + parsed["timedelta"]

            self.levels.add(parsed)
            self.markers.check(parsed)
        else:
            parsed = content[-1]
            parsed["message"] += f"\n{line}"

        return parsed

    def stream(self):
        """
        Streams new lines as they come in. If multiple new lines are retrieved at once,
        returns them as a group
        """
        self.levels.reset()
        self.markers.reset()
        lines = self._load(self.file)

        new = []
        for line in lines:
            if line:
                self.parse(line, new)
            else:
                yield new
                new = []

    def read(self):
        """
        Reads the entirety of the log file as it presently exists
        """
        return next(self.stream())


class Unknown(FileFinder):
    extensions = ["*"]
    patterns = {r"(.*)": "Directory unknown, unable to determine this file"}

    def _load(self, *args, **kwargs):
        """
        Files under this class are ignored
        """
        self.log.error(
            "Unable to load file as the parent directory was unable to be parsed"
        )


class IsofitWD(FileFinder):
    extensions = ["*"]
    patterns = {
        r"(config)": "ISOFIT configuration files",
        r"(data)": "Additional data files generated by ISOFIT",
        r"(input)": "Data files input to ISOFIT",
        r"(lut)": "Look-Up-Table outputs",
        r"(lut_full)": "Look-Up-Table outputs",
        r"(lut_h2o)": "Look-Up-Table outputs for a presolve run",
        r"(output)": "ISOFIT outputs such as reflectance",
    }
    classes = {
        "config": Config,
        "data": Data,
        "input": Input,
        "lut": LUT,
        "output": Output,
    }

    def __init__(self, *args, recursive=True, **kwargs):
        """
        Parameters
        ----------
        recursive : bool, default=True
            If a sub directory type cannot be determined, recursively use the IsofitWD
            class to instantiate on that directory. This enables finding multiple valid
            IsofitWD under a path. If set to False, will use the `Unknown` class
            instead which disables most functionality for the given directory
        """
        # This class should not be saving to cache because
        # it defers loading to child classes which have their own cache
        kwargs["cache"] = False

        super().__init__(*args, **kwargs)

        # No path provided, will load current dir lazily on first call
        if self.path is None:
            return

        self.logs = Logs(self.path)

        self.dirs = {}

        unkn = []
        dirs = [file.name for file in self.path.glob("*") if file.is_dir()]
        for subdir in sorted(dirs):
            for name, cls in self.classes.items():
                if name in subdir:
                    self.log.debug(f"Initializing {subdir} with class {cls.__name__}")
                    self.dirs[subdir] = cls(self.path / subdir)
                    break
            else:
                unkn.append(subdir)

        alt = Unknown
        if recursive:
            alt = IsofitWD

        for subdir in unkn:
            self.log.debug(f"Initializing {subdir} with class {alt.__class__.__name__}")
            self.dirs[subdir] = alt(self.path / subdir)

    def __getattr__(self, key):
        # Auto reset to the current working directory if the object was initialized without an input
        if self.__dict__.get("path") is None:
            self.reset(".")
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self.dirs:
            return self.dirs[key]

    def _load(self, file):
        parent, subpath = self.subpath(file, parent=True)

        if parent in self.dirs:
            return self.dirs[parent].load(path=subpath)
        else:
            self.log.warning(f"Files on the root path are not supported at this time: {subpath}")

    def reset(self, *args, **kwargs):
        """
        Re-initializes the object

        Parameters
        ----------
        *args : list
            Arguments to pass directly to __init__
        **kwargs : dict
            Key-word arguments to pass directly to __init__

        Returns
        -------
        self : IsofitWD
            Re-initialized IsofitWD object
        """
        self.__init__(*args, **kwargs)
        return self

    def subpath(self, path, parent=False):
        """
        Converts an absolute path to a relative path under self.path

        Parameters
        ----------
        path : str
            Either absolute or relative path, will assert it exists under self.path
        parent : bool, default=False
            Split the top-level parent from the subpath

        Returns
        -------
        pathlib.Path | (str, pathlib.Path)
            Relative path
            If parent is enabled, returns the top-level parent separated from the path
        """
        path = Path(path)

        if path.is_absolute():
            path = path.relative_to(self.path)

        if not (file := self.path / path).exists():
            raise FileNotFoundError(file)

        if parent:
            parent = "."
            if len(path.parents) >= 2:
                # -1 = "./" == self.path
                # -2 = "./dir/"
                parent = path.parents[-2].name

            return parent, path.relative_to(parent)
        return path

    def info(self, file):
        """
        Overrides the inherited info function to pass the file to the correct child
        object's info function

        Parameters
        ----------
        file : str
            File name to compare against the patterns dict keys

        Returns
        -------
        any
            Returns the value if a regex key in the patterns dict matches the file name
        """
        parent, subpath = self.subpath(file, parent=True)

        if parent in self.dirs:
            return self.dirs[parent].info(subpath.name)
        return super().info(file)

    def getTree(self, info=False, **kwargs):
        """
        Recursively finds the files under a directory as a dict tree

        Overrides the inherited getTree function to call the getTree of every object
        in self.dirs and merge the returns together. This lets each child handle
        building its own tree

        Parameters
        ----------
        info : bool, default=False
            Return the found files as objects with their respective info

        Returns
        -------
        tree : list
            Tree structure of discovered files
        """
        tree = []

        # First handle known subdirectories (mapped to handlers)
        for name, obj in self.dirs.items():
            if info:
                name = FileInfo(name, self.info(name))

            tree.append({name: obj.getTree(info=info, **kwargs)})

        # Now scan the actual directory to catch unknown entries
        try:
            with os.scandir(self.path) as scan:
                for path in scan:
                    if (name := path.name) not in self.dirs:
                        if info:
                            name = FileInfo(name, self.info(name))
                        tree.append(name)
        except:
            self.log.exception(f"Error reading path {path}")

        return tree
