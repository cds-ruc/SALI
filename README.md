# SALI (Updating)



Thanks for all of you for your interest in our work.

SALI, A Scalable Adaptive Learned Index Framework based on Probability Models, includes node-evolving strategies and a lightweight statistics maintenance strategy, aiming to achieve high scalability, improve efficiency, and enhance the robustness of the learned index.

This project contains the code of SALI and welcomes contributions or suggestions.

## Compile & Run

```bash
mkdir build
cd build
cmake ..
make
```

- Run example:

```bash
./build/example_mt
```


To enable the evolving strategy, please modify the following parameters in the 'src/core/sali.h':

- Enabling read-evolving:

```bash
#define READ_EVOLVE 1
```


- Enabling space compression:

```bash
#define COMPRESS 1
```

Note that there is no need to disable insert evolving, just like in learned indexes where the retraining does not need to be turned off.


## Usage

`src/examples/example_multithread.cpp` shows how to use sali in multithread environment.

## Running benchmark


SALI's performance can be assessed using the GRE benchmarking tool. We have integrated SALI into GRE as "GRE_SALI," which is a fork of GRE, and you can access it at the following link: https://github.com/YunWorkshop/GRE_SALI.

Note that for additional features, we have introduced additional flags in GRE_SALI:

- Enable skewed workload for lookup:

```bash
--sample_distribution=zipf
```

- Enable skewed workload for insert:

```bash
--hot_write=1
```

Other configurations, such as workloads and datasets, are left to your discretion for evaluation. For more details on these configurations, please refer to [GRE](https://github.com/YunWorkshop/GRE_SALI).


## Acknowledgements

- The index structure chosen to evaluate the effectiveness of SALI is LIPP, and our implementation is based on the code of [LIPP](https://github.com/Jiacheng-WU/lipp).
- The benchmark used to evaluate this paper is [GRE](https://github.com/gre4index/GRE).

