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
