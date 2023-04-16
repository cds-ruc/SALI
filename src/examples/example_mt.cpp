#include <iostream>
#include <sali.h>

#include "omp.h"

using namespace std;

int main() {
  sali::SALI<int, int> sali;

  int key_num = 5;
  pair<int, int> *keys = new pair<int, int>[key_num];
  omp_set_num_threads(4);

// #pragma omp parallel for schedule(static, 1)
  /*for (int i = 0; i < key_num; i++) {
     keys[i] = {i, i};
  }*/
  // printf("bulk loading\n");
  keys[0]={1,1};
  keys[1]={10,10};
  keys[2]={100,100};
  keys[3]={1000,1000};
  keys[4]={10000,10000};
  sali.bulk_load(keys, 5);

#pragma omp parallel for schedule(static, 4)
  for (int i = 0; i < 1000; i++) {
    sali.insert(i,i);
  }

  // show tree structure
  // sali.show();

  return 0;
}
