/* Unit 2/4: correlated risk and SPD solve using real C kernels. */
#include "hft/api.h"
#include <stdio.h>
int main(void) {
  double covariance[] = {1, .2, .2, 2}, weights[] = {.5, .5}, var, cvar,
         x[2] = {0}, residual;
  int iterations;
  if (!hft_cg(covariance, weights, 2, x, 32, 1e-10, 0, &iterations,
              &residual) ||
      !hft_monte_carlo(covariance, weights, 2, 10000, 42, &var, &cvar, 0))
    return 1;
  printf("CG residual %.12g VaR99 %.6f CVaR99 %.6f\n", residual, var, cvar);
  return 0;
}
