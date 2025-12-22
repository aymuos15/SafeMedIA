# Types of Differential Privacy (DP)

This project implements two layers of Differential Privacy in a Federated Learning context.

## 1. Sample-level DP (Local)
* **Goal**: Protect individual data points (e.g., a single medical image).
* **Implementation**: Client-side using **Opacus**.
* **Mechanism**: Clips gradients and adds noise to model updates before they leave the client.

## 2. User-level DP (Global)
* **Goal**: Protect the participation of the user/client itself.
* **Implementation**: Server-side using **Flower**.
* **Mechanism**: Clips aggregated model updates and adds noise to mask the contribution of any single client.

## 3. Hybrid DP
* **Goal**: Maximum privacy protection.
* **Implementation**: Combines both Sample-level and User-level DP.
* **Accounting**: Tracked independently but composed using Renyi Differential Privacy (RDP).
