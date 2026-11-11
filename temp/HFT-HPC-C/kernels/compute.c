/* OpenCL C device entry points. Each simulation work item owns a pair; no
 * atomics or cross-workgroup barriers are required. Unit 3. */
__kernel void layout(__global ulong *sizes) {
  sizes[0] = sizeof(HftSymbol);
  sizes[1] = sizeof(HftConfig);
  sizes[2] = sizeof(HftModel);
}
__kernel void simulate(__global HftSymbol *states, __global const HftConfig *c,
                       int groups, int offset, ulong steps, ulong delay) {
  int id = (int)get_global_id(0) + offset;
  if (id < groups)
    hft_group_inline(states, id * 2, c->symbols, c, steps, 0, delay);
}
__kernel void covariance(__global const float *x, __global float *out, int rows,
                         int cols, int tiled) {
  int i = (int)get_global_id(0), j = (int)get_global_id(1);
  if (i >= cols || j >= cols)
    return;
  double a = 0, b = 0, v = 0;
  for (int k = 0; k < rows; ++k) {
    a += x[k * cols + i];
    b += x[k * cols + j];
  }
  a /= rows;
  b /= rows;
  if (tiled) {
    for (int block = 0; block < rows; block += 32)
      for (int k = block; k < rows && k < block + 32; ++k)
        v += (x[k * cols + i] - a) * (x[k * cols + j] - b);
  } else
    for (int k = 0; k < rows; ++k)
      v += (x[k * cols + i] - a) * (x[k * cols + j] - b);
  out[i * cols + j] = (float)(v / (rows - 1));
}
__kernel void inference(__global const HftModel *m, __global const float *x,
                        __global float *y, int rows) {
  int i = (int)get_global_id(0);
  if (i < rows) {
    float p[3];
    hft_infer_shared(m, x + i * 32, p);
    for (int j = 0; j < 3; ++j)
      y[i * 3 + j] = p[j];
  }
}
__kernel void gradient(__global const HftModel *m, __global const float *x,
                       __global const int *y, __global float *g, int rows,
                       int start) {
  int i = (int)get_global_id(0);
  if (i < rows) {
    for (int w = 0; w < HFT_WEIGHTS; ++w)
      g[i * HFT_WEIGHTS + w] = 0;
    hft_gradient_shared(m, x + (start + i) * 32, y[start + i],
                        g + i * HFT_WEIGHTS);
  }
}
__kernel void update(__global HftModel *m, __global const float *g, int rows) {
  int w = (int)get_global_id(0);
  if (w < HFT_WEIGHTS) {
    float sum = 0;
    for (int i = 0; i < rows; ++i)
      sum += g[i * HFT_WEIGHTS + w];
    m->weights[w] -= .005f * (sum / rows + .0001f * m->weights[w]);
  }
}
__kernel void qlearning(__global HftSymbol *states, __global HftConfig *configs,
                        int episodes, int steps) {
  int e = (int)get_global_id(0);
  if (e >= episodes)
    return;
  __global HftConfig *c = configs + e;
  __global HftSymbol *s = states + e * 2;
  float q[HFT_QSTATES * 9];
  for (int j = 0; j < HFT_QSTATES * 9; ++j)
    q[j] = c->model.q[j];
  for (int i = 0; i < steps; ++i) {
    HftTop t = hft_top(&s->book, c, 0);
    int state = hft_q_state(s, t, c), act = 0;
    HftRandom r = hft_random(c->seed, 0, 811, (ulong)i);
    for (int a = 1; a < 9; ++a)
      if (q[state * 9 + a] > q[state * 9 + act])
        act = a;
    if (hft_uniform(r.x[0]) < .2)
      act = r.x[1] % 9;
    long old = s->position.cash + (long)s->position.qty * hft_mid(t);
    for (int j = 0; j < HFT_QSTATES * 9; ++j)
      c->model.q[j] = q[j];
    for (int a = 0; a < 9; ++a)
      c->model.q[state * 9 + a] = a == act ? 1e20f : 0;
    hft_group_inline(s, 0, 2, c, 1, 0, 0);
    t = hft_top(&s->book, c, 0);
    int next = hft_q_state(s, t, c);
    float best = q[next * 9];
    for (int a = 1; a < 9; ++a)
      if (q[next * 9 + a] > best)
        best = q[next * 9 + a];
    double reward =
        (double)(s->position.cash + (long)s->position.qty * hft_mid(t) - old) -
        .001 * s->position.qty * s->position.qty;
    q[state * 9 + act] +=
        .05f * ((float)reward + .95f * best - q[state * 9 + act]);
  }
  for (int j = 0; j < HFT_QSTATES * 9; ++j)
    c->model.q[j] = q[j];
}
/* A workgroup reuses two 8x32 tiles in local memory. All work items reach both
 * barriers, including padded columns. O(T*N^2), O(512 floats) local storage. */
__kernel void covariance_tiled(__global const float *x, __global float *out,
                               int rows, int cols, int ignored) {
  __local float left[8][32], right[8][32];
  int lx = (int)get_local_id(0), ly = (int)get_local_id(1),
      i = (int)get_global_id(0), j = (int)get_global_id(1),
      linear = ly * 8 + lx;
  double a = 0, b = 0, v = 0;
  for (int base = 0; base < rows; base += 32) {
    for (int idx = linear; idx < 256; idx += 64) {
      int col = idx / 32, k = idx % 32, ci = (int)get_group_id(0) * 8 + col,
          cj = (int)get_group_id(1) * 8 + col;
      left[col][k] =
          base + k < rows && ci < cols ? x[(base + k) * cols + ci] : 0;
      right[col][k] =
          base + k < rows && cj < cols ? x[(base + k) * cols + cj] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < 32 && base + k < rows; ++k) {
      double u = left[lx][k], w = right[ly][k];
      a += u;
      b += w;
      v += u * w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (i < cols && j < cols)
    out[i * cols + j] = (float)((v - a * b / rows) / (rows - 1));
  (void)ignored;
}
