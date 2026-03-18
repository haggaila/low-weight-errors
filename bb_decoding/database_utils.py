# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Database utilities for managing decoder and simulation data.

This module provides functions to save and load decoder data, simulation results,
and maintain CSV databases for tracking experiments. Uses pickle for data serialization
and pandas for database management.
"""

import csv
import pickle
import os.path
import pandas as pd
from typing import Dict, List, Optional, Any, Union

S_OUTPUT_FOLDER = "bb_output"
"""File name to use for the database of all decoder data files."""

S_DECODER_DATA_DB_FILENAME = "decoder.database.csv"
"""File name to use for the database of all decoder data files."""

S_DECODER_DATA_PREFIX = "decoder"
"""Prefix for the file names of all decoder data files."""

S_SIMULATION_DB_FILENAME = "simulation.database.csv"
"""File name to use for the database of all stochastic error simulation result files."""

S_DETECTOR_DB_FILENAME = "detector.database.csv"
"""File name to use for the database of all direct detectors decoding simulation result files."""

S_SIMULATION_PREFIX = "simulation"
"""Prefix for the file names of all simulation result files."""


def generate_decoding_paths(n_up_dirs=0):
    """
    Create and return paths for output directories.
    
    Generates paths for output, decoder data, simulation results, and figures.
    Creates directories if they don't exist.

    Args:
        n_up_dirs: Number of parent directories to traverse (default: 0).

    Returns:
        A 4-tuple: (output_path, decoder_path, simulation_path, plot_path).
    """
    s_output_path = ""
    for _ in range(n_up_dirs):
        s_output_path += "../"
    s_output_path = (
        os.path.abspath(s_output_path + S_OUTPUT_FOLDER) + "/"
    )  # use the absolute path of the current file
    if not os.path.exists(s_output_path):
        os.mkdir(s_output_path)
    s_decoder_path = s_output_path + "decoders/"
    if not os.path.exists(s_decoder_path):
        os.mkdir(s_decoder_path)
    s_simulation_path = s_output_path + "simulations/"
    if not os.path.exists(s_simulation_path):
        os.mkdir(s_simulation_path)
    s_plot_path = s_output_path + "figures/"
    if not os.path.exists(s_plot_path):
        os.mkdir(s_plot_path)
    return (
        s_output_path,
        s_decoder_path,
        s_simulation_path,
        s_plot_path,
    )


def save_to_db(s_db_path: str, line_data: dict):
    """
    Append a data record to a CSV database file.
    
    Creates the database file with headers if it doesn't exist, then appends
    the new record using pandas.

    Args:
        s_db_path: Full path to the CSV database file.
        line_data: Dictionary containing the data record to save.
    """
    if not os.path.isfile(s_db_path):
        # New database, write header line based on metadata keys
        with open(s_db_path, "w") as f:
            header = line_data.keys()
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

    db_line = {}
    for key in line_data.keys():
        db_line[key] = [line_data[key]]
    line_df = pd.DataFrame(db_line)
    df = pd.read_csv(s_db_path)
    if df.empty:
        df = line_df
    else:
        df = pd.concat([df, line_df])
    df.to_csv(s_db_path, index=False)


def save_decoder_data(db_line: Dict, decoder_data: Dict):
    """
    Save decoder data to a pickle file and update the decoder database.
    
    Args:
        db_line: Metadata dictionary to save in the CSV database.
        decoder_data: Complete decoder data structure to pickle.
    """
    s_uuid = db_line["unique_id"]
    s_output_path, s_decoder_path, _, _ = generate_decoding_paths()

    s_data_filename = s_decoder_path + S_DECODER_DATA_PREFIX + "." + s_uuid + ".pkl"
    print("Saving data to ", s_data_filename)
    with open(s_data_filename, "wb") as fp:
        pickle.dump(decoder_data, fp)

    s_db_path = s_output_path + S_DECODER_DATA_DB_FILENAME
    save_to_db(s_db_path, db_line)


def load_decoder_data(s_uuid: str) -> Dict:
    """
    Load decoder data from a pickle file by UUID.
    
    Args:
        s_uuid: Unique identifier for the decoder data.
        
    Returns:
        Dictionary containing the decoder data.
    """
    _, s_decoder_path, _, _ = generate_decoding_paths()
    s_data_filename = s_decoder_path + S_DECODER_DATA_PREFIX + "." + s_uuid + ".pkl"
    print("Loading decoder data from ", s_data_filename)
    with open(s_data_filename, "rb") as fp:
        decoder_data = pickle.load(fp)
    return decoder_data


def load_simulation_data(s_uuid: str, n_up_dirs=2) -> Dict:
    """
    Load simulation results from a pickle file by UUID.
    
    Args:
        s_uuid: Unique identifier for the simulation data.
        n_up_dirs: Number of parent directories to traverse (default: 2).
        
    Returns:
        Dictionary containing the simulation results.
    """
    _, _, s_simulation_path, _ = generate_decoding_paths(n_up_dirs=n_up_dirs)
    s_data_filename = s_simulation_path + S_SIMULATION_PREFIX + "." + s_uuid + ".pkl"
    print("Loading simulation data from ", s_data_filename)
    with open(s_data_filename, "rb") as fp:
        sim_data = pickle.load(fp)
    return sim_data


def save_simulation_data(
    db_line: Dict, simulation_results: Dict, s_simulation_db_filename: str
):
    """
    Save simulation results to a pickle file and update the simulation database.
    
    Args:
        db_line: Metadata dictionary to save in the CSV database.
        simulation_results: Complete simulation results to pickle.
        s_simulation_db_filename: Name of the simulation database CSV file.
    """
    s_uuid = db_line["unique_id"]
    s_output_path, _, s_simulation_path, _ = generate_decoding_paths()

    s_data_filename = s_simulation_path + S_SIMULATION_PREFIX + "." + s_uuid + ".pkl"
    print("Saving data to ", s_data_filename)
    with open(s_data_filename, "wb") as fp:
        pickle.dump(simulation_results, fp)

    s_db_path = s_output_path + s_simulation_db_filename
    save_to_db(s_db_path, db_line)


def query_simulations_by_name(
    s_simulations_name: str,
    sort_by: Optional[Any] = None,
    ascending: Union[bool, List[bool]] = True,
    na_position="last",
    parse_dates=None,
    n_up_dirs=2,
    s_db_filename=S_SIMULATION_DB_FILENAME,
):
    """
    Query simulations from the database by name and return filtered results.

    Args:
        s_simulations_name: Name field to filter by (empty string returns all).
        sort_by: Column(s) to sort by (uses pandas sort_values).
        ascending: Sort order (ascending or descending).
        na_position: Position of NaN values in sorting ('last' or 'first').
        parse_dates: Columns to parse as dates.
        n_up_dirs: Number of parent directories to traverse (default: 2).
        s_db_filename: Database filename to query.

    Returns:
        Pandas DataFrame with filtered and sorted simulation records.
    """

    s_output_path, _, _, _ = generate_decoding_paths(n_up_dirs=n_up_dirs)
    s_db_path = s_output_path + s_db_filename

    df = pd.read_csv(s_db_path, parse_dates=parse_dates, keep_default_na=False)
    if s_simulations_name != "":
        df_2 = df.loc[df["name"] == s_simulations_name]
    else:
        df_2 = df
    if sort_by is not None:
        df_3 = df_2.sort_values(sort_by, ascending=ascending, na_position=na_position)
    else:
        df_3 = df_2
    return df_3


def find_simulation_id(files: List[str], s_filter_query: str):
    """
    Find simulation IDs matching a query across multiple database files.

    Args:
        files: List of CSV database file paths to search.
        s_filter_query: Pandas query string to filter simulations.

    Returns:
        List of unique simulation IDs matching the query.
    """
    selected_id = []
    for file in files:
        df = pd.read_csv(file)
        df1 = df.query(s_filter_query)
        selected_id.extend(df1.id.unique())
    return selected_id
