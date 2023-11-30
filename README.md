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


To enable the evolving strategy, please modify the following parameters in `src/core/sali.h`:

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

`src/examples/example_multithread.cpp` demonstrates the usage of SALI in a multithreaded environment:


```bash
#include <iostream>
#include <sali.h>
#include "omp.h"

using namespace std;

int main() {
  sali::SALI<int, int> sali;

  int key_num = 1000;
  pair<int, int> *keys = new pair<int, int>[key_num];
  for (int i = 0; i < 1000; i++) {
    keys[i]={i,i};
  }
  sali.bulk_load(keys, 1000);

  omp_set_num_threads(12);

#pragma omp parallel for schedule(static, 12)
  for (int i = 1000; i < 2000; i++) {
    sali.insert(i,i);
  }
#pragma omp parallel for schedule(static, 12)
  for (int i = 0; i < 2000; i++) {
    std::cout<<"value at "<<i<<": "<<sali.at(i)<<std::endl;
  }

  return 0;
}
```

## Running benchmark


SALI's performance can be assessed using the GRE benchmarking tool. We have integrated SALI into GRE as "[GRE_SALI](https://github.com/YunWorkshop/GRE_SALI)," which is a fork of GRE. In GRE_SALI, you can assess the performance of SALI comprehensively.

Note that for additional features, we have introduced additional flags in GRE_SALI:

- Enable skewed workload for lookup:

```bash
--sample_distribution=zipf
```

- Enable skewed workload for insert:

```bash
--hot_write=true
```

Other configurations, such as workloads and datasets, are left to your discretion for evaluation. For more details on these configurations, please refer to [GRE](https://github.com/YunWorkshop/GRE_SALI).


## License

This project is licensed under the terms of the MIT License.


## Acknowledgements

- The LIPP structure is chosen to evaluate the effectiveness of SALI framework, and our implementation is based on the code of [LIPP](https://github.com/Jiacheng-WU/lipp).
- SALI utilize the PLA algorithm of the [PGM](https://github.com/gvinciguerra/PGM-index), which is used to construct the PGM. 
- The benchmark used to evaluate the SALI is [GRE](https://github.com/gre4index/GRE).
