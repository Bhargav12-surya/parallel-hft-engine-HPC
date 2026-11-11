/* Unit 2: run and compare actual serial / parallel C results. */
#include "hft/api.h"
#include <stdio.h>
int main(void) {
  HftConfig c;
  HftResult a = {0}, b = {0};
  hft_config_default(&c);
  c.symbols = 8;
  c.events = 1000;
  c.engine = HFT_SERIAL;
  if (!hft_run(&c, &a, NULL, NULL, NULL))
    return 1;
  c.engine = HFT_THREADS;
  if (!hft_run(&c, &b, NULL, NULL, NULL)) {
    hft_result_free(&a);
    return 1;
  }
  int same = hft_result_equivalent(&a, &b);
  printf("serial %.6fs threaded %.6fs equal=%d\n", a.seconds, b.seconds, same);
  hft_result_free(&a);
  hft_result_free(&b);
  return same ? 0 : 1;
}
