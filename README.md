# EdgeSWA-FPGA: Sliding-Window Attention Accelerator on PYNQ-Z1

A prototype HLS implementation of fixed-point sliding-window attention with approximate SoftMax for FPGA-based edge inference on the PYNQ-Z1/Zynq-7020 platform.

<p align="center">
  <img src="figures/system_architecture.png" width="850">
</p>

## Overview

Transformer self-attention provides strong modeling capability, but its computational cost grows quadratically with sequence length:

$$
\mathcal{O}(N^2D)
$$

where:

- \(N\) is the sequence length,
- \(D\) is the feature dimension.

This quadratic complexity makes conventional full attention difficult to deploy on small embedded FPGA platforms with limited DSPs, BRAM, and memory bandwidth.

This project implements **causal sliding-window attention**, where each token attends only to a fixed number of nearby tokens. Its computational complexity becomes:

\[
O(NWD)
\]

where \(W\) is the sliding-window size.

The proposed accelerator combines:

- fixed-point sliding-window attention,
- input-stationary query reuse,
- piecewise approximate SoftMax,
- AXI-based PS–PL communication,
- physical deployment on a PYNQ-Z1 board.

> **Main contribution:** A working fixed-point sliding-window attention accelerator with approximate SoftMax was deployed and evaluated on physical FPGA hardware.

---

## Motivation

Full self-attention compares every query token with every key token:

\[
S = QK^T
\]

This produces an \(N \times N\) attention matrix and requires approximately:

\[
O(N^2D)
\]

operations.

This becomes challenging on an embedded FPGA because:

- the attention matrix grows quadratically,
- Q, K, and V transfers create substantial memory traffic,
- exact exponential and division operations in SoftMax are expensive,
- the available LUT, DSP, BRAM, and DDR bandwidth are limited.

Sliding-window attention reduces the number of comparisons by restricting each query token to a local window.

For token \(i\), the implemented causal window is:

\[
j \in [\max(0,i-W+1),i]
\]

Therefore, each token attends only to itself and its preceding nearby tokens.

---

## Research Questions

This prototype investigates four questions:

1. **Feasibility:** Can fixed-point sliding-window attention fit and execute on the PYNQ-Z1 programmable logic?

2. **Approximation impact:** How closely does the FPGA output match floating-point and approximate software references?

3. **Performance:** How does FPGA execution latency compare with an ARM CPU software implementation?

4. **Scalability:** How do latency, throughput, and output error change as the sequence length increases from \(N=16\) to \(N=128\)?

---

## Contributions

This project provides:

- An HLS implementation of single-head causal sliding-window attention.
- An input-stationary dataflow that reuses each query vector while streaming its local key and value window.
- INT8 input and output processing.
- Piecewise approximations for SoftMax exponential and reciprocal operations.
- AXI-Lite control communication between the processing system and accelerator.
- AXI HP access to Q, K, V, and output tensors stored in DDR.
- Physical FPGA evaluation on PYNQ-Z1.
- Evaluation of correctness, latency, throughput, speedup, resource utilization, timing, and estimated power.

---

## System Architecture

The system contains three primary components:

### Processing System

The ARM-based processing system performs:

- Linux and Jupyter execution,
- physically contiguous buffer allocation,
- input generation,
- Q, K, V, and output address configuration,
- accelerator start and completion control,
- output collection and performance measurement.

### Programmable Logic

The programmable logic contains the HLS attention accelerator and performs:

- query-vector loading,
- local key/value-window loading,
- query-key dot products,
- score scaling,
- approximate SoftMax,
- weighted value accumulation,
- INT8 output saturation.

### DDR Memory

The DDR memory stores:

- query matrix \(Q\),
- key matrix \(K\),
- value matrix \(V\),
- output matrix \(O\).

The processing system controls the accelerator through **AXI-Lite**, while the programmable logic accesses tensors through the **AXI HP DDR interface**.

<p align="center">
  <img src="figures/ps_pl_architecture.png" width="850">
</p>

---

## Sliding-Window Attention Algorithm

For every query token \(Q_i\), the accelerator processes only the keys and values inside its causal window.

The first valid key index is:

\[
j_{\min}=\max(0,i-W+1)
\]

The attention scores are calculated as:

\[
s_{ij}
=
\frac{Q_iK_j^T}{\sqrt{D}},
\qquad
j_{\min}\leq j\leq i
\]

The normalized attention output is:

\[
O_i
=
\sum_{j=j_{\min}}^{i}
\alpha_{ij}V_j
\]

where:

\[
\alpha_{ij}
=
\frac{\exp(s_{ij}-s_{\max})}
{\sum_k \exp(s_{ik}-s_{\max})}
\]

and:

\[
s_{\max}=\max_j s_{ij}
\]

Unlike full attention, scores outside the local window are never computed.

---

## Hardware Dataflow

The accelerator follows an input-stationary query dataflow.

For each token \(i\):

1. Load query vector \(Q_i\) into an on-chip buffer.
2. Determine the causal sliding-window boundaries.
3. Stream the corresponding \(K_j\) and \(V_j\) vectors.
4. Compute query-key dot products.
5. identify the maximum attention score.
6. Apply score shifting and clipping.
7. Approximate the exponential values.
8. Approximate the reciprocal of the SoftMax denominator.
9. Normalize the attention weights.
10. Accumulate the weighted value vectors.
11. Saturate and store the INT8 output.

<p align="center">
  <img src="figures/kernel_dataflow.png" width="850">
</p>

---

## Approximate SoftMax

Exact SoftMax requires exponential and division operations:

\[
\alpha_j
=
\frac{\exp(s_j-s_{\max})}
{\sum_k \exp(s_k-s_{\max})}
\]

These operations are expensive on a small FPGA.

The accelerator therefore uses the following approximation procedure:

1. Subtract the maximum score for numerical stability.
2. Clip the shifted score to the range \([-8,0]\).
3. Replace the exact exponential with piecewise constant approximations.
4. Accumulate the approximate exponential values.
5. Replace exact division with reciprocal approximation buckets.
6. Normalize the attention weights using the approximate reciprocal.
7. Saturate the final output to the INT8 range.

The approximate Python reference reproduces this behavior and is used to distinguish hardware implementation error from approximation error.

---

## Implementation Configuration

| Parameter | Value |
|---|---:|
| FPGA board | PYNQ-Z1 |
| FPGA device | Zynq-7020 |
| HLS tool | Vitis HLS |
| FPGA integration | Vivado 2025.2 |
| Target clock | 100 MHz |
| Attention type | Single-head causal SWA |
| Input type | INT8 |
| Output type | INT8 |
| Feature dimension \(D\) | 16 |
| Window size \(W\) | 4 |
| Sequence lengths \(N\) | 16, 32, 64, 128 |

---

## Repository Structure

```text
EdgeSWA-FPGA/
├── README.md
├── LICENSE
├── hls/
│   ├── sliding_attention.cpp
│   ├── sliding_attention.hpp
│   ├── testbench.cpp
│   └── run_hls.tcl
├── vivado/
│   ├── block_design/
│   ├── constraints/
│   └── build_project.tcl
├── pynq/
│   ├── sliding_attention.bit
│   ├── sliding_attention.hwh
│   └── run_accelerator.ipynb
├── software/
│   ├── float_reference.py
│   ├── approximate_reference.py
│   ├── generate_inputs.py
│   └── evaluate_results.py
├── results/
│   ├── accuracy_results.csv
│   ├── latency_results.csv
│   ├── utilization_report.txt
│   ├── timing_report.txt
│   └── power_report.txt
├── figures/
│   ├── system_architecture.png
│   ├── ps_pl_architecture.png
│   ├── kernel_dataflow.png
│   ├── accuracy_results.png
│   ├── latency_results.png
│   └── resource_utilization.png
└── docs/
    └── project_presentation.pdf
