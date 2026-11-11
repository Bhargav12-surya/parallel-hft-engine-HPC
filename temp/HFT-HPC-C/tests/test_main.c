/* Behavioral regression gates, C17 only. No framework dependency. */
#include "hft/api.h"
#include "hft/backprop.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int checks = 0, failures = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(x)) {                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);             \
      ++failures;                                                              \
    }                                                                          \
  } while (0)
static void books(void) {
  HftConfig c;
  hft_config_default(&c);
  c.symbols = 2;
  c.fee = 1;
  HftSymbol *s = calloc(1, sizeof(*s));
  hft_book_clear(&s->book, &c);
  HftOrder o = {1, 0, 10002, 1, 5, 1, 0, 0};
  CHECK(hft_submit(s, &c, o, 0, 0));
  o.id = 2;
  o.ts = 2;
  o.qty = 7;
  CHECK(hft_submit(s, &c, o, 0, 0));
  o = (HftOrder){99, 0, 10002, 3, 6, 0, 5, 1};
  CHECK(hft_submit(s, &c, o, 0, 0));
  CHECK(!s->book.nodes[1].live);
  CHECK(s->book.nodes[2].order.qty == 6);
  CHECK(s->position.qty == 6);
  CHECK(s->position.cash == -60018);
  CHECK(s->book.levels[HFT_LEVELS + 258].volume == 6);
  CHECK(hft_best(&s->book, &c, 1, 0) == hft_best(&s->book, &c, 1, 1));
  CHECK(hft_book_cancel(&s->book, &c, 2));
  CHECK(!hft_best(&s->book, &c, 1, 0));
  CHECK(!hft_book_cancel(&s->book, &c, 2));
  free(s);
}
static void math_tests(void) {
  HftRandom r = hft_random(
      0, 0, 0, 0); /* Random123 Philox zero counter/key known answer. */
  CHECK(r.x[0] == 0x6627e8d5u);
  CHECK(r.x[1] == 0xe169c58du);
  for (int i = 1; i < 300; ++i) {
    double x = i * .05;
    CHECK(fabs(hft_log(x) - log(x)) < 1e-10);
    CHECK(fabs(hft_sqrt(x) - sqrt(x)) < 1e-10);
    CHECK(fabs(hft_exp(x) - exp(x)) < 1e-8 * fmax(1, exp(x)));
  }
  double a[4] = {4, 1, 1, 3}, b[2] = {1, 2}, x[2] = {0}, l[4], res;
  int it;
  CHECK(hft_cholesky(a, 2, l));
  CHECK(hft_cg(a, b, 2, x, 10, 1e-10, 0, &it, &res));
  CHECK(fabs(x[0] - 1.0 / 11) < 1e-10);
  CHECK(fabs(x[1] - 7.0 / 11) < 1e-10);
  double data[6] = {1, 2, 2, 4, 3, 6}, cov[4];
  CHECK(hft_covariance(data, 3, 2, cov, 0, 0));
  CHECK(cov[0] == 1 && cov[1] == 2 && cov[3] == 4);
}
static void queues(void) {
  HftQueue q = {0};
  for (int i = 0; i < 63; ++i)
    CHECK(hft_queue_push(&q, i));
  int v;
  for (int i = 0; i < 63; ++i) {
    CHECK(hft_queue_pop(&q, &v));
    CHECK(v == i);
  }
  CHECK(!hft_queue_pop(&q, &v));
  HftTripleBuffer *b = calloc(1, sizeof(*b));
  HftSnapshot *s = calloc(1, sizeof(*s));
  b->back = 2;
  b->front = 0;
  hft_store(&b->middle, 1);
  for (int i = 0; i < 100; ++i) {
    s->pnl = i;
    hft_snapshot_publish(b, s);
    if (i % 4 == 0) {
      CHECK(hft_snapshot_consume(b, s));
      CHECK(s->pnl == i);
    }
  }
  free(s);
  free(b);
  HftHistogram *h = calloc(1, sizeof(*h));
  for (int i = 1; i <= 1000; ++i)
    hft_hist_record(h, (hft_u64)i);
  CHECK(hft_hist_percentile(h, .99) >= 990 &&
        hft_hist_percentile(h, .99) <= 995);
  free(h);
}
static void engines(void) {
  HftConfig c;
  hft_config_default(&c);
  c.symbols = 6;
  c.events = 300;
  c.batch = 17;
  c.threads = 3;
  for (int strategy = 0; strategy < 13; ++strategy) {
    c.strategy = strategy;
    c.engine = HFT_NAIVE;
    HftResult a = {0}, b = {0};
    CHECK(hft_run(&c, &a, NULL, NULL, NULL));
    for (int engine = HFT_SERIAL; engine <= HFT_OPENMP; ++engine) {
      HftCapability cap;
      hft_probe(engine, &cap);
      if (!cap.available)
        continue;
      c.engine = engine;
      CHECK(hft_run(&c, &b, NULL, NULL, NULL));
      CHECK(hft_result_equivalent(&a, &b));
      if (!hft_result_equivalent(&a, &b))
        fprintf(stderr, "strategy %d engine %d hashes %llx %llx\n", strategy,
                engine, (unsigned long long)a.hash, (unsigned long long)b.hash);
      for (int i = 0; i < c.symbols; ++i)
        CHECK(abs(b.states[i].position.qty) <= c.max_position);
      hft_result_free(&b);
    }
    hft_result_free(&a);
  }
  c.engine = HFT_SERIAL;
  char error[HFT_ERROR] = {0};
  CHECK(hft_config_save(&c, "test-config.json", error));
  HftConfig d;
  CHECK(hft_config_load(&d, "test-config.json", error));
  CHECK(c.seed == d.seed && c.events == d.events && c.strategy == d.strategy);
  remove("test-config.json");
  HftResult a = {0}, b = {0};
  CHECK(hft_run(&c, &a, NULL, NULL, NULL));
  CHECK(hft_result_export(&a, "test-result", error));
  CHECK(hft_result_import(&b, "test-result.json", error));
  CHECK(hft_result_equivalent(&a, &b));
  hft_result_free(&a);
  hft_result_free(&b);
  remove("test-result.json");
  remove("test-result.csv");
}
static void ai(void) {
  HftModel m;
  hft_model_init(&m, 42);
  float x[32], p[3], g[HFT_WEIGHTS] = {0};
  for (int i = 0; i < 32; ++i)
    x[i] = .01f * i;
  hft_mlp_predict(&m, x, p);
  CHECK(fabs(p[0] + p[1] + p[2] - 1) < 1e-6);
  double loss = hft_mlp_gradient(&m, x, 2, g);
  int indices[] = {0, 33, 2088, 2112, 4159, 4192, 4289};
  for (int i = 0; i < 7; ++i) {
    int w = indices[i];
    float v = m.weights[w];
    m.weights[w] = v + .001f;
    hft_mlp_predict(&m, x, p);
    double plus = -log(p[2]);
    m.weights[w] = v - .001f;
    hft_mlp_predict(&m, x, p);
    double minus = -log(p[2]);
    m.weights[w] = v;
    CHECK(fabs((plus - minus) / .002 - g[w]) < .003);
  }
  CHECK(isfinite(loss));
  char error[HFT_ERROR];
  CHECK(hft_model_save(&m, "test-model.bin", error));
  HftModel other;
  CHECK(hft_model_load(&other, "test-model.bin", error));
  CHECK(!memcmp(&m, &other, sizeof(m)));
  remove("test-model.bin");
}
int main(void) {
  books();
  math_tests();
  queues();
  ai();
  engines();
  printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
