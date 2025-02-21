import os
import io
import zipfile
import requests
import datetime
import numpy as np
import pandas as pd
from typing import Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "data"))
MEM_DUMPSTER_DTYPE_DICT = {
    "timestamp_ms": np.float64,
    "hash": str,
    "gas": np.float64,
    "gas_price": np.float64,
    "gas_tip_cap": np.float64,
    "gas_fee_cap": np.float64,
    "data_size": np.float64,
    "sources": str,
    "included_at_block_height": np.int64,
    "included_block_timestamp_ms": np.float64,
    "inclusion_delay_ms": np.float64,
}


def gather_tx_features_for_day_range(
    start_date_str: str,
    end_date_str: str,
    data_dir: str = DATA_DIR,
    block_data_file: str = "eth_blocks_gas_6_months.csv",
    block_lags: int = 1,
    checkpoint: bool = True,
    return_df: bool = False,
    sample_frac: float = 1.0,
) -> Optional[pd.DataFrame]:
    start_date_dt = pd.to_datetime(start_date_str)
    end_date_dt = pd.to_datetime(end_date_str)
    days = int((end_date_dt - start_date_dt).days)
    features_list = []
    for d in range(days):
        day_dt = start_date_dt + datetime.timedelta(days=d)
        day_str = day_dt.strftime("%Y-%m-%d")
        print(f" ")
        print(f"Processing features for {day_str}...")
        day_features_df = gather_tx_features_for_day(
            day_str, data_dir, block_data_file, block_lags
        )
        if sample_frac < 1.0:
            day_features_df = day_features_df.sample(frac=sample_frac)
        if checkpoint:
            out_dir = os.path.join(data_dir, "features")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_file = os.path.join(out_dir, f"{day_str}.csv")
            day_features_df.to_csv(out_file, index=False)
        if return_df:
            features_list.append(day_features_df)
    if return_df:
        features_df = pd.concat(features_list, ignore_index=True)
        return features_df
    else:
        return None


def gather_tx_features_for_day(
    day_str: str = "2025-01-01",
    data_dir: str = DATA_DIR,
    block_data_file: str = "eth_blocks_gas_6_months.csv",
    block_lags: int = 1,
) -> pd.DataFrame:
    # Load mempool dumpster data for main day
    day_tx_df = load_mempool_dumpster_data(day_str, data_dir)
    day_tx_hashes = day_tx_df[["hash"]]
    # Add additional mempool dumpster data for 2h before and after
    tx_df = complement_mempool_dumpster_data(day_tx_df, day_str, data_dir, pad_hours=2)
    # Load block data and combine
    blocks_df = load_block_data(block_data_file, data_dir)
    tx_bl_df = combine_tx_and_block_dfs(tx_df, blocks_df, block_lags)
    # Add aggregated mempool data
    tx_bl_mem_df = add_mempool_aggregated_features(tx_bl_df)
    # Keep only txs from day_str and sort
    features_df = (
        day_tx_hashes.merge(tx_bl_mem_df, how="inner", on="hash")
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )
    return features_df


def load_block_data(file_name: str, data_dir: str) -> pd.DataFrame:
    # Read csv file
    file_dir = os.path.join(data_dir, file_name)
    blocks_df = pd.read_csv(file_dir)
    # Format datetime column
    blocks_df["timestamp"] = pd.to_datetime(blocks_df["timestamp"]).dt.tz_localize(None)
    return blocks_df


def load_mempool_dumpster_data(day_str: str, data_dir: str) -> pd.DataFrame:
    # Read csv file
    file_path = os.path.join(data_dir, "mempool_dumpster", day_str + ".csv")
    if not os.path.isfile(file_path):
        # if file has not been downloaded, then download it first
        download_mempool_dumpster_csv(day_str, data_dir)
    tx_df = pd.read_csv(file_path, dtype=MEM_DUMPSTER_DTYPE_DICT)
    # Filter data -> only "local" source
    tx_df = tx_df[tx_df["sources"].str.contains("local")]
    # Format datetime columns
    tx_df["arrival_time"] = pd.to_datetime(tx_df["timestamp_ms"], unit="ms")
    tx_df["included_block_timestamp_ms"] = pd.to_datetime(
        tx_df["included_block_timestamp_ms"], unit="ms"
    )
    # Select relevant columns and sort by arrival time
    select_cols = [
        "arrival_time",
        "hash",
        "gas",
        "gas_tip_cap",
        "gas_fee_cap",
        "data_size",
        "inclusion_delay_ms",
        "included_at_block_height",
    ]
    tx_df = tx_df[select_cols].sort_values("arrival_time")
    return tx_df


def download_mempool_dumpster_csv(day_str: str, data_dir: str = DATA_DIR) -> None:
    print(f"Downloading data for {day_str}...")
    month_str = day_str[:-3]
    zip_file_url = f"https://mempool-dumpster.flashbots.net/ethereum/mainnet/{month_str}/{day_str}.csv.zip"
    r = requests.get(zip_file_url)
    if r.ok:
        print("Download complete!")
    else:
        raise Exception("Download unsuccessful :(")
    # Get zip file and decompress into data_dir
    z = zipfile.ZipFile(io.BytesIO(r.content))
    out_dir = os.path.join(data_dir, "mempool_dumpster")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    z.extractall(out_dir)


def complement_mempool_dumpster_data(
    day_tx_df: pd.DataFrame, day_str: str, data_dir: str, pad_hours: int = 2
) -> pd.DataFrame:
    # Load mempool dumpster data for 2h after
    next_day_dt = pd.to_datetime(day_str) + datetime.timedelta(days=1)
    next_day_str = next_day_dt.strftime("%Y-%m-%d")
    next_df = load_mempool_dumpster_data(next_day_str, data_dir)
    next_hours_dt = pd.to_datetime(day_str) + datetime.timedelta(days=1, hours=2)
    next_df = next_df[next_df["arrival_time"] < next_hours_dt]
    # Load mempool dumpster data for 2h before
    prev_day_dt = pd.to_datetime(day_str) - datetime.timedelta(days=1)
    prev_day_str = prev_day_dt.strftime("%Y-%m-%d")
    prev_df = load_mempool_dumpster_data(prev_day_str, data_dir)
    prev_hours_dt = pd.to_datetime(day_str) - datetime.timedelta(hours=2)
    prev_df = prev_df[prev_df["arrival_time"] > prev_hours_dt]
    # Add additional hours of data to main day and sort
    padded_day_tx_df = pd.concat([day_tx_df, next_df, prev_df], ignore_index=True)
    padded_day_tx_df = padded_day_tx_df.sort_values("arrival_time")
    return padded_day_tx_df


def combine_tx_and_block_dfs(
    tx_df: pd.DataFrame, blocks_df: pd.DataFrame, block_lags: int
) -> pd.DataFrame:
    # Compute previous block height and add to tx data
    block_times = blocks_df[["timestamp", "block_number"]]
    block_times.columns = ["prev_block_time", "prev_block_height"]
    tx_bl_df = pd.merge_asof(
        tx_df,
        block_times,
        left_on="arrival_time",
        right_on="prev_block_time",
        direction="backward",
    )
    # Compute transaction delay in blocks
    tx_bl_df["inclusion_delay_blocks"] = (
        tx_bl_df["included_at_block_height"] - tx_bl_df["prev_block_height"]
    )
    # Exclude txs with non-positive delays
    tx_bl_df = tx_bl_df[tx_bl_df["inclusion_delay_blocks"] > 0].reset_index(drop=True)
    # Add info on previous blocks since tx arrival (one for each lag level)
    for lag in range(block_lags):
        lagged_blocks = blocks_df.copy()
        lagged_blocks["block_number"] = lagged_blocks["block_number"] + lag
        tx_bl_df = tx_bl_df.merge(
            lagged_blocks,
            how="left",
            left_on="prev_block_height",
            right_on="block_number",
        )
        tx_bl_df = tx_bl_df.drop(columns=["timestamp", "block_number"])
        tx_bl_df = tx_bl_df.rename(
            columns={
                "size_bytes": f"lag_{lag+1}_block_size_bytes",
                "gas_used": f"lag_{lag+1}_block_gas_used",
                "blob_gas_used": f"lag_{lag+1}_block_blob_gas_used",
                "transaction_count": f"lag_{lag+1}_block_tx_count",
                "base_fee_gwei": f"lag_{lag+1}_block_base_fee_gwei",
            }
        )
    return tx_bl_df


def add_mempool_aggregated_features(tx_bl_df: pd.DataFrame) -> pd.DataFrame:
    mem_agg_df = compute_aggregated_mempool_features(tx_bl_df)
    tx_bl_mem_df = tx_bl_df.merge(
        mem_agg_df, how="left", left_on="prev_block_height", right_on="block_height"
    ).drop(columns="block_height")
    return tx_bl_mem_df


def compute_aggregated_mempool_features(tx_bl_df: pd.DataFrame) -> pd.DataFrame:
    # Select relevant columns
    cols = [
        "hash",
        "gas",
        "gas_tip_cap",
        "gas_fee_cap",
        "data_size",
        "prev_block_height",
        "inclusion_delay_blocks",
        "included_at_block_height",
    ]
    mem_df = tx_bl_df[cols]
    # Repeat transaction info for each block the transaction remains in the mempool
    mem_df = mem_df.iloc[
        mem_df.index.repeat(mem_df["inclusion_delay_blocks"])
    ].reset_index(drop=True)
    # Add block heights for each block the transaction remains in the mempool
    mem_df["in_mempool_at_height"] = np.concat(
        [
            np.arange(start, end)
            for start, end in zip(
                tx_bl_df["prev_block_height"], tx_bl_df["included_at_block_height"]
            )
        ]
    )
    # Compute aggregated features by block slot
    mem_agg_df = mem_df.groupby("in_mempool_at_height")[["gas", "data_size"]].sum()
    mem_agg_df["tx_count"] = mem_df.groupby("in_mempool_at_height").size()
    gas_fee_stats_df = (
        mem_df.groupby("in_mempool_at_height")["gas_fee_cap"]
        .quantile([0.1, 0.5, 0.9])
        .unstack()
    )
    gas_tip_stats_df = (
        mem_df.groupby("in_mempool_at_height")["gas_tip_cap"]
        .quantile([0.1, 0.5, 0.9])
        .unstack()
    )
    mem_agg_df = pd.concat(
        [mem_agg_df, gas_fee_stats_df, gas_tip_stats_df], axis=1
    ).reset_index()
    mem_agg_df.columns = (
        ["block_height", "mem_total_gas_limit", "mem_total_data_size", "mem_tx_count"]
        + [f"mem_gas_fee_cap_{c}" for c in gas_fee_stats_df.columns]
        + [f"mem_gas_tip_cap_{c}" for c in gas_tip_stats_df.columns]
    )
    return mem_agg_df


if __name__ == "__main__":
    df = gather_tx_features_for_day(
        day_str="2025-01-01",
        data_dir=DATA_DIR,
        block_data_file="eth_blocks_gas_6_months.csv",
        block_lags=5,
    )
    out_file = os.path.join(DATA_DIR, "debug_output.csv")
    df.to_csv(out_file, index=False)
