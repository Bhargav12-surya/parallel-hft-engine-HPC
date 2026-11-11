/* Unit 1: inspect fixed data layout and generate one indexed random counter. */
#include "hft/api.h"
#include "hft/numeric.h"
#include <stdio.h>
int main(void) {
  HftRandom r = hft_random(42, 1, 7, 100);
  printf("event=%zu bytes symbol=%zu bytes config=%zu bytes random=%08x\n",
         sizeof(HftEvent), sizeof(HftSymbol), sizeof(HftConfig), r.x[0]);
  return 0;
}
