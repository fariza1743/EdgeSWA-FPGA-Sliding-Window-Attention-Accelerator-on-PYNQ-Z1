# EdgeSWA-FPGA: Sliding-Window Attention Accelerator on PYNQ-Z1

A prototype HLS implementation of fixed-point sliding-window attention with approximate SoftMax for FPGA-based edge inference on the PYNQ-Z1/Zynq-7020 platform.

This work was done as a project in "Hardware for Neural Network" course (PhD).

## Overview

Transformer self-attention provides strong modeling capability, but its computational cost grows quadratically with sequence length:

$$
\mathcal{O}(N^2D)
$$

where:

- N is the sequence length,
- D is the feature dimension.

This quadratic complexity makes conventional full attention difficult to deploy on small embedded FPGA platforms with limited DSPs, BRAM, and memory bandwidth.

This project implements **causal sliding-window attention**, where each token attends only to a fixed number of nearby tokens. Its computational complexity becomes:

$$
\mathcal{O}{(NWD)}
$$

where W is the sliding-window size.

The proposed accelerator combines:

- Fixed-point sliding-window attention,
- Input-stationary query reuse,
- Piecewise approximate SoftMax,
- AXI-based PS–PL communication,
- Physical deployment on a PYNQ-Z1 board.

> **Main contribution:** A working fixed-point sliding-window attention accelerator with approximate SoftMax was deployed and evaluated on physical FPGA hardware.

---

## Motivation

Full self-attention compares every query token with every key token:

$$
\mathcal{S} = {QK^T}
$$

This produces an $$ (N \times N\) $$ attention matrix and requires approximately:

$$
\mathcal{O}(N^2D)
$$
operations.

This becomes challenging on an embedded FPGA because:

- The attention matrix grows quadratically,
- Q, K, and V transfers create substantial memory traffic,
- Exact exponential and division operations in SoftMax are expensive,
- The available LUT, DSP, BRAM, and DDR bandwidth are limited.

Sliding-window attention reduces the number of comparisons by restricting each query token to a local window.

For token $$\(i\)$$, the implemented causal window is:

$$
j \in [\max(0,i-W+1),i]
$$

Therefore, each token attends only to itself and its preceding nearby tokens.

---

## Research Questions

This prototype investigates four questions:

1. **Feasibility:** Can fixed-point sliding-window attention fit and execute on the PYNQ-Z1 programmable logic?

2. **Approximation impact:** How closely does the FPGA output match floating-point and approximate software references?

3. **Performance:** How does FPGA execution latency compare with an ARM CPU software implementation?

4. **Scalability:** How do latency, throughput, and output error change as the sequence length increases from N=16 to N=128 ?

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

- query matrix Q,
- key matrix K,
- value matrix V,
- output matrix O.

The processing system controls the accelerator through **AXI-Lite**, while the programmable logic accesses tensors through the **AXI HP DDR interface**.



## Sliding-Window Attention Algorithm

For each query token $Q_i$, the accelerator processes only the keys and values within its causal attention window.

The first valid key index is

$$
j_{\min} = \max(0, i - W + 1),
$$

where $W$ is the sliding-window size.

The attention score between query $Q_i$ and key $K_j$ is computed as

$$
s_{ij} = \frac{Q_i K_j^{T}}{\sqrt{D}},
\qquad
j_{\min} \leq j \leq i,
$$

where $D$ is the query and key dimension.

For numerical stability, the maximum attention score within the current window is

\max_{j_{\min} \leq k \leq i} s_{ik}.
$$

The normalized attention weights are then computed as

\frac{
\exp\left(s_{ij}-s_{\max}^{(i)}\right)
}{
\displaystyle
\sum_{k=j_{\min}}^{i}
\exp\left(s_{ik}-s_{\max}^{(i)}\right)
},
\qquad
j_{\min} \leq j \leq i.
$$

Finally, the attention output for query token $Q_i$ is

\sum_{j=j_{\min}}^{i}
\alpha_{ij}V_j.
$$

Unlike full self-attention, attention scores for tokens outside the local causal window are never computed. Therefore, the computational complexity is reduced from

$$
\mathcal{O}(N^2D)
$$

to approximately

$$
\mathcal{O}(NWD),
$$

where $N$ is the sequence length, $W$ is the sliding-window size, and $D$ is the feature dimension.



## Hardware Dataflow

The accelerator follows an input-stationary query dataflow, in which each query vector remains in an on-chip buffer while the corresponding key and value vectors are streamed through the processing units.

For each query token $i$, the accelerator performs the following steps:

Load the query vector $Q_i$ into an on-chip buffer.
Determine the boundaries of the causal sliding window.
Stream the corresponding key vectors $K_j$ and value vectors $V_j$.
Compute the query-key dot products.
Identify the maximum attention score within the window.
Apply score shifting and clipping for numerical stability.
Approximate the exponential values.
Approximate the reciprocal of the SoftMax denominator.
Normalize the attention weights.
Accumulate the weighted value vectors.
Apply saturation and store the resulting INT8 output.


## Approximate SoftMax

Exact SoftMax requires exponential and division operations:

\frac{\exp\left(s_j-s_{\max}\right)}
{\displaystyle\sum_k \exp\left(s_k-s_{\max}\right)}.
$$

These operations are computationally expensive and resource-intensive on a small FPGA.

To reduce hardware complexity, the accelerator applies the following approximation procedure:

Subtract the maximum attention score for numerical stability.
Clip the shifted score to the range $[-8, 0]$.
Replace the exact exponential function with a piecewise-constant approximation.
Accumulate the approximate exponential values to form the SoftMax denominator.
Replace exact division with a bucket-based reciprocal approximation.
Normalize the attention weights using the approximate reciprocal.
Accumulate the weighted value vectors and saturate the final output to the INT8 range.

A Python reference implementation reproduces the same approximate SoftMax behavior. It is used to separate two sources of error:

Approximation error: the difference between exact SoftMax and the approximate Python implementation.
Hardware implementation error: the difference between the approximate Python implementation and the RTL accelerator output.

This comparison helps verify whether observed numerical differences are caused by the approximation method itself or by an error in the hardware implementation.

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
