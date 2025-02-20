-- This query was run on Google Big Query
-- https://console.cloud.google.com/bigquery?inv=1&invt=Abp-yA&project=ethereum-public-data-407811&ws=!1m0
-- It collects gas information from Ethereum raw block data
-- Query run on Feb. 17th, 2025
SELECT
  timestamp,
  number AS block_number,
  size AS size_bytes,
  gas_used,
  blob_gas_used,
  transaction_count,
  base_fee_per_gas/100000000 AS base_fee_gwei
FROM
  bigquery-public-data.crypto_ethereum.blocks
WHERE
  timestamp >= CAST(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH) AS TIMESTAMP)
ORDER BY
  block_number