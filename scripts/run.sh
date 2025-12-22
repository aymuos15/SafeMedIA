#!/bin/bash
# DP-FedMed run script for all DP styles

# Function to run simulation with a specific config
run_sim() {
  local config_file=$1
  echo "=========================================================="
  echo "Running simulation with: $config_file"
  echo "=========================================================="

  flwr run . \
    --run-config "config-file=\"$config_file\""
}

# Run all 4 styles one by one
run_sim "configs/none.toml"
run_sim "configs/sample.toml"
run_sim "configs/user.toml"
run_sim "configs/default.toml" # Hybrid
