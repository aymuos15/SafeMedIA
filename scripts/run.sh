#!/bin/bash
# DP-FedMed run script with federation configuration

flwr run . \
  --federation-config 'options.num-supernodes=2' \
  --federation-config 'options.backend.client-resources.num-cpus=1' \
  --federation-config 'options.backend.client-resources.num-gpus=0.3'
