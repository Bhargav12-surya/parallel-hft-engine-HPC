/* Native desktop entirely in C: Win32/GDI or X11. Simulation/training/benchmark
 * jobs run on a service thread; immutable snapshots cross a triple buffer.
 * UI allocations/rendering never run in a timed simulation event. */
#include "hft/api.h"
#include "kernels.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#endif
#define BG 0x0b1220
#define PANEL 0x121e30
#define BORDER 0x24354b
#define FG_TEXT 0xe6edf5
#define MUTED 0x91a3bb
#define CYAN 0x45d6c6
#define RED 0xef6c7b
#define GOLD 0xefbd68
typedef struct {
  int x, y, w, h, id;
} Hit;
typedef struct {
  HftConfig config;
  HftResult result, working;
  HftModel trained_model;
  int model_ready;
  double pending_loss, pending_accuracy, pending_reward;
  HftAtomic stop, done, pause;
  HftThread thread;
  HftTripleBuffer triple;
  HftSnapshot snapshot;
  int busy, job, page, selected, network, presentation, lesson;
  double loss, accuracy, reward;
  char message[HFT_ERROR];
  Hit hits[160];
  int nhits, w, h;
  HftFill tape[32];
  int tape_count;
  double last_update;
} App;
static App app;
static int headless;
static unsigned char *pixels;
#include "hft/font.h"
static void pixel(int x, int y, unsigned c) {
  if (x < 0 || y < 0 || x >= app.w || y >= app.h)
    return;
  size_t i = ((size_t)y * app.w + x) * 3;
  pixels[i] = (unsigned char)(c >> 16);
  pixels[i + 1] = (unsigned char)(c >> 8);
  pixels[i + 2] = (unsigned char)c;
}
static void soft_rect(int x, int y, int w, int h, unsigned c) {
  for (int j = y; j < y + h && j < app.h; ++j)
    for (int i = x; i < x + w && i < app.w; ++i)
      pixel(i, j, c);
}
static void soft_line(int x1, int y1, int x2, int y2, unsigned c) {
  int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1, dy = -abs(y2 - y1),
      sy = y1 < y2 ? 1 : -1, err = dx + dy;
  for (;;) {
    pixel(x1, y1, c);
    if (x1 == x2 && y1 == y2)
      break;
    int twice = 2 * err;
    if (twice >= dy) {
      err += dy;
      x1 += sx;
    }
    if (twice <= dx) {
      err += dx;
      y1 += sy;
    }
  }
}
static void soft_text(int x, int y, const char *s, unsigned c, int large) {
  int scale = large ? 2 : 1;
  for (; *s; ++s) {
    unsigned ch = (unsigned char)*s;
    if (ch < 32 || ch > 126)
      ch = 63;
    for (int j = 0; j < 18; ++j)
      for (int i = 0; i < 9; ++i)
        if (hft_font[ch - 32][j] & (1u << i))
          soft_rect(x + i * scale, y + j * scale, scale, scale, c);
    x += 9 * scale;
  }
}

#ifdef _WIN32
static HWND window;
static HDC dc;
static HFONT font, heading;
static COLORREF color(unsigned c) {
  return RGB((c >> 16) & 255, (c >> 8) & 255, c & 255);
}
static void rect(int x, int y, int w, int h, unsigned c) {
  if (headless) {
    soft_rect(x, y, w, h, c);
    return;
  }
  RECT r = {x, y, x + w, y + h};
  HBRUSH b = CreateSolidBrush(color(c));
  FillRect(dc, &r, b);
  DeleteObject(b);
}
static void line(int x1, int y1, int x2, int y2, unsigned c) {
  if (headless) {
    soft_line(x1, y1, x2, y2, c);
    return;
  }
  HPEN p = CreatePen(PS_SOLID, 1, color(c));
  HGDIOBJ old = SelectObject(dc, p);
  MoveToEx(dc, x1, y1, NULL);
  LineTo(dc, x2, y2);
  SelectObject(dc, old);
  DeleteObject(p);
}
static void textat(int x, int y, const char *s, unsigned c, int large) {
  if (headless) {
    soft_text(x, y, s, c, large);
    return;
  }
  SelectObject(dc, large ? heading : font);
  SetTextColor(dc, color(c));
  SetBkMode(dc, TRANSPARENT);
  TextOutA(dc, x, y, s, (int)strlen(s));
}
#else
static Display *display;
static Window window;
static GC gc;
static Pixmap canvas;
static XFontStruct *font;
static void rect(int x, int y, int w, int h, unsigned c) {
  if (headless) {
    soft_rect(x, y, w, h, c);
    return;
  }
  if (w <= 0 || h <= 0)
    return;
  XSetForeground(display, gc, c);
  XFillRectangle(display, canvas, gc, x, y, (unsigned)w, (unsigned)h);
}
static void line(int x1, int y1, int x2, int y2, unsigned c) {
  if (headless) {
    soft_line(x1, y1, x2, y2, c);
    return;
  }
  XSetForeground(display, gc, c);
  XDrawLine(display, canvas, gc, x1, y1, x2, y2);
}
static void textat(int x, int y, const char *s, unsigned c, int large) {
  if (headless) {
    soft_text(x, y, s, c, large);
    return;
  }
  (void)large;
  XSetForeground(display, gc, c);
  XDrawString(display, canvas, gc, x, y + 14, s, (int)strlen(s));
}
#endif
static void label(int x, int y, unsigned colorcode, const char *format,
                  double value) {
  char s[256];
  snprintf(s, sizeof(s), format, value);
  textat(x, y, s, colorcode, 0);
}
static void button(int x, int y, int w, const char *title, int id, int active) {
  rect(x, y, w, 32, active ? 0x214e57 : PANEL);
  line(x, y + 31, x + w, y + 31, active ? CYAN : BORDER);
  textat(x + 10, y + 7, title, active ? CYAN : FG_TEXT, 0);
  if (app.nhits < 160)
    app.hits[app.nhits++] = (Hit){x, y, w, 32, id};
}
static void card(int x, int y, int w, const char *title, const char *value,
                 unsigned c) {
  rect(x, y, w, 86, PANEL);
  textat(x + 16, y + 12, title, MUTED, 0);
  textat(x + 16, y + 41, value, c, 1);
}
static void info(int x, int y, const char *s) { textat(x, y, s, MUTED, 0); }
static void progress(const HftSnapshot *s, void *user) {
  App *a = user;
  hft_snapshot_publish(&a->triple, s);
  while (hft_load(&a->pause) && !hft_load(&a->stop))
    hft_sleep(20);
  if (a->job == 1)
    hft_sleep(4);
}
static void *service(void *user) {
  App *a = user;
  HftConfig c = a->config;
  int ok = 1;
  a->message[0] = 0;
  if (a->job == 1) {
    if (c.engine == HFT_MPI || c.engine == HFT_HYBRID)
      ok = hft_run_mpi_process(&c, &a->working, &a->stop, progress, a,
#ifdef _WIN32
                               "hft_mpi_worker.exe",
#else
                               "./hft_mpi_worker",
#endif
                               c.threads);
    else
      ok = hft_run(&c, &a->working, &a->stop, progress, a);
    if (!ok)
      snprintf(a->message, sizeof(a->message), "%s", a->working.error);
    else
      snprintf(a->message, sizeof(a->message),
               "Completed %llu events in %.3f seconds",
               (unsigned long long)a->working.events, a->working.seconds);
  } else if (a->job == 2) {
    HftDataset d;
    if (hft_dataset_make(&c, c.train_samples, &d, a->message)) {
      if (c.engine == HFT_OPENCL) {
        double seconds;
        ok = hft_opencl_train(&c.model, d.features, d.labels, d.train_count,
                              c.train_epochs, &seconds, a->message);
      } else
        ok = hft_mlp_train(&c.model, &d, c.train_epochs, c.threads,
                           &a->pending_loss, &a->pending_accuracy, &a->stop);
      hft_dataset_free(&d);
      if (ok) {
        a->trained_model = c.model;
        a->model_ready = 1;
        snprintf(a->message, sizeof(a->message),
                 "MLP trained. Loss %.4f; validation accuracy %.1f%%",
                 a->pending_loss, a->pending_accuracy * 100);
      }
    }
  } else if (a->job == 5) {
    // Real‑data loading and training
    HftDataset d;
    if (hft_real_dataset_load(app.config.real_data_path, &d, a->message)) {
        if (c.engine == HFT_OPENCL) {
            double seconds;
            ok = hft_opencl_train(&c.model, d.features, d.labels, d.train_count,
                                  c.train_epochs, &seconds, a->message);
        } else {
            ok = hft_mlp_train(&c.model, &d, c.train_epochs, c.threads,
                               &a->pending_loss, &a->pending_accuracy, &a->stop);
        }
        hft_dataset_free(&d);
        if (ok) {
            a->trained_model = c.model;
            a->model_ready = 1;
            snprintf(a->message, sizeof(a->message),
                     "Real‑data MLP trained. Loss %.4f; accuracy %.1f%%",
                     a->pending_loss, a->pending_accuracy * 100);
        }
    }
  } else if (a->job == 3) {
    ok = hft_q_train(&c.model, &c, 16, 512, c.threads, &a->pending_reward,
                     &a->stop);
    if (ok) {
      a->trained_model = c.model;
      a->model_ready = 1;
      snprintf(a->message, sizeof(a->message),
               "Q-learning trained: 16 episodes; mean penalized reward %.2f",
               a->pending_reward);
    }
  } else if (a->job == 4) {
    hft_mkdir("results");
    for (int scenario = 1; scenario <= 8 && !hft_load(&a->stop); ++scenario) {
      HftSnapshot s = {0};
      snprintf(s.status, sizeof(s.status), "Benchmark S%d / 8", scenario);
      s.progress = (scenario - 1) / 8.0;
      hft_snapshot_publish(&a->triple, &s);
      if (scenario == 8) {
        HftCapability cap;
        hft_probe(HFT_MPI, &cap);
        if (cap.available) {
          HftConfig mc = c;
          mc.engine = HFT_MPI;
          HftResult mr = {0};
          hft_run_mpi_process(&mc, &mr, &a->stop, progress, a,
#ifdef _WIN32
                              "hft_mpi_worker.exe",
#else
                              "./hft_mpi_worker",
#endif
                              2);
          if (mr.events)
            hft_result_export(&mr, "results/S8-mpi", a->message);
          hft_result_free(&mr);
        }
        continue;
      }
      char scen[8], events[32], symbols[24];
      snprintf(scen, sizeof(scen), "S%d", scenario);
      snprintf(events, sizeof(events), "%llu", (unsigned long long)c.events);
      snprintf(symbols, sizeof(symbols), "%d", scenario == 1 ? 1 : c.symbols);
      char *args[] = {"hft_bench",
                      "--scenario",
                      scen,
                      "--events",
                      events,
                      "--symbols",
                      symbols,
                      "--engines",
                      "serial_naive,serial_opt,threaded,openmp,opencl",
                      "--warmup",
                      "0",
                      "--reps",
                      "1",
                      "--out",
                      "results"};
      hft_bench_main(17, args);
    }
    snprintf(a->message, sizeof(a->message),
             "Suite finished. Open results/ for measured CSV, JSON and SVG "
             "reports.");
  }
  hft_store(&a->done, 1);
  return NULL;
}
static void start(int job) {
  if (app.busy)
    return;
  hft_result_free(&app.working);
  memset(&app.working, 0, sizeof(app.working));
  app.model_ready = 0;
  app.job = job;
  hft_store(&app.stop, 0);
  hft_store(&app.done, 0);
  hft_store(&app.pause, 0);
  app.busy = hft_thread_start(&app.thread, service, &app);
  if (!app.busy)
    strcpy(app.message, "Could not start background service");
}
static void save_screenshot(void) {
  int w = app.w, h = app.h;
  unsigned char *rgb = malloc((size_t)w * h * 3);
  if (!rgb)
    return;
#ifdef _WIN32
  HDC screen = GetDC(window), memory = CreateCompatibleDC(screen);
  BITMAPINFO bi = {0};
  bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bi.bmiHeader.biWidth = w;
  bi.bmiHeader.biHeight = -h;
  bi.bmiHeader.biPlanes = 1;
  bi.bmiHeader.biBitCount = 32;
  bi.bmiHeader.biCompression = BI_RGB;
  void *pixels;
  HBITMAP bitmap =
      CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &pixels, NULL, 0);
  HGDIOBJ old = SelectObject(memory, bitmap);
  BitBlt(memory, 0, 0, w, h, screen, 0, 0, SRCCOPY);
  for (int i = 0; i < w * h; ++i) {
    unsigned char *p = (unsigned char *)pixels + 4 * i;
    rgb[3 * i] = p[2];
    rgb[3 * i + 1] = p[1];
    rgb[3 * i + 2] = p[0];
  }
  SelectObject(memory, old);
  DeleteObject(bitmap);
  DeleteDC(memory);
  ReleaseDC(window, screen);
#else
  XImage *image = XGetImage(display, window, 0, 0, (unsigned)w, (unsigned)h,
                            AllPlanes, ZPixmap);
  if (!image) {
    free(rgb);
    return;
  }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      unsigned long p = XGetPixel(image, x, y);
      size_t i = ((size_t)y * w + x) * 3;
      rgb[i] = (unsigned char)(p >> 16);
      rgb[i + 1] = (unsigned char)(p >> 8);
      rgb[i + 2] = (unsigned char)p;
    }
  XDestroyImage(image);
#endif
  hft_mkdir("results");
  hft_write_png("results/studio.png", rgb, w, h, app.message);
  free(rgb);
}
static void click(int id) {
  if (id >= 100 && id < 108) {
    app.page = id - 100;
    return;
  }
  if (id == 1) {
    start(1);
    return;
  }
  if (id == 2) {
    hft_store(&app.pause, !hft_load(&app.pause));
    return;
  }
  if (id == 3) {
    hft_store(&app.stop, 1);
    hft_store(&app.pause, 0);
    return;
  }
  if (id == 4) {
    if (!app.busy) {
      hft_result_free(&app.result);
      memset(&app.result, 0, sizeof(app.result));
      memset(&app.snapshot, 0, sizeof(app.snapshot));
      strcpy(app.message, "Reset. Configuration retained.");
    }
    return;
  }
  if (id == 5) {
    save_screenshot();
    return;
  }
  if (app.busy)
    return;
  if (id >= 200 && id < 213)
    app.config.strategy = id - 200;
  else if (id >= 300 && id < 308) {
    HftCapability cap;
    hft_probe(id - 300, &cap);
    if (cap.available && id - 300 != HFT_NVIDIA)
      app.config.engine = id - 300;
    snprintf(app.message, sizeof(app.message), "%s: %s", cap.name, cap.reason);
  } else if (id == 10) {
    hft_mkdir("results");
    if (app.result.events) {
      hft_result_export(&app.result, "results/desktop", app.message);
      hft_svg_report(&app.result, "results/report.svg", app.message);
    }
    hft_config_save(&app.config, "results/config.json", app.message);
    hft_model_save(&app.config.model, "results/model.bin", app.message);
    save_screenshot();
    strcpy(
        app.message,
        "Report pack saved in results/: JSON, CSV, SVG, PNG, config, model.");
  } else if (id == 11) {
    HftResult r = {0};
    if (hft_result_import(&r, "results/desktop.json", app.message)) {
      hft_result_free(&app.result);
      app.result = r;
      strcpy(app.message, "Imported results/desktop.json");
    }
  } else if (id == 12)
    start(2);
  else if (id == 13)
    start(3);
  else if (id == 14)
    start(4);
  else if (id == 20)
    app.config.symbols = app.config.symbols >= 256 ? 2 : app.config.symbols * 2;
  else if (id == 21)
    app.config.events =
        app.config.events >= 100000 ? 1000 : app.config.events * 10;
  else if (id == 22)
    app.config.threads = app.config.threads >= 16 ? 1 : app.config.threads * 2;
  else if (id == 23)
    app.config.batch = app.config.batch >= 1024 ? 1 : app.config.batch * 4;
  else if (id == 24)
    app.config.regime = (app.config.regime + 1) % 7;
  else if (id == 25)
    app.config.auto_regime = !app.config.auto_regime;
  else if (id == 26)
    app.config.mode = !app.config.mode;
  else if (id == 27) {
    app.network = (app.network + 1) % 4;
    double ns[] = {0, 50000, 500000, 20000000};
    app.config.network_ns = ns[app.network];
  } else if (id == 28)
    app.config.latency_amplification =
        app.config.latency_amplification >= 100
            ? 1
            : app.config.latency_amplification * 10;
  else if (id == 29)
    app.config.cpu_hourly =
        app.config.cpu_hourly >= 2 ? .05 : app.config.cpu_hourly * 2;
  else if (id == 30)
    app.config.fast = app.config.fast >= 16 ? 2 : app.config.fast * 2;
  else if (id == 31)
    app.config.slow = app.config.slow >= 128 ? 32 : app.config.slow * 2;
  else if (id == 32)
    app.config.quantity =
        app.config.quantity >= 16 ? 1 : app.config.quantity * 2;
  else if (id == 33)
    app.config.max_position =
        app.config.max_position >= 400 ? 100 : app.config.max_position * 2;
  else if (id == 34)
    app.config.volatility =
        app.config.volatility >= .04 ? .0025 : app.config.volatility * 2;
  else if (id == 35)
    app.config.schedule = (app.config.schedule + 1) % 3;
  else if (id == 36)
    app.config.seed++;
  else if (id == 37)
    app.config.mode = 1, app.page = 1, start(1);
  else if (id == 38)
    app.lesson = (app.lesson + 1) % 8;
  else if (id == 39)
    app.lesson = app.lesson ? app.lesson - 1 : 7;
  else if (id == 40) {
    hft_mkdir("results");
    hft_source_export("results/shared-engine.c", app.message);
    strcpy(app.message, "Exported the actual shared C/OpenCL source.");
  } else if (id == 41) {
    app.presentation = !app.presentation;
    app.page = 0;
  } else if (id == 42) {
    if (hft_model_load(&app.config.model, "results/model.bin", app.message))
      strcpy(app.message, "Loaded results/model.bin");
  } else if (id == 43) {
    if (hft_config_load(&app.config, "results/config.json", app.message))
      strcpy(app.message, "Loaded results/config.json");
  }
}
static void plot(int x, int y, int w, int h, int candles) {
  HftSnapshot *s = &app.snapshot;
  rect(x, y, w, h, PANEL);
  for (int k = 1; k < 5; ++k)
    line(x, y + k * h / 5, x + w, y + k * h / 5, BORDER);
  int n = s->count;
  if (!candles && !app.busy && app.result.point_count)
    n = app.result.point_count;
  if (n < 2) {
    info(x + 24, y + 40, "Run a simulation to populate this chart.");
    return;
  }
  double low = 1e30, high = -1e30;
  for (int i = 0; i < n; ++i) {
    int at = (s->head - s->count + i + HFT_TRACE) % HFT_TRACE;
    double a, b;
    if (candles) {
      a = (double)s->candles[at].low;
      b = (double)s->candles[at].high;
    } else {
      a = b = !app.busy && app.result.point_count
                  ? (double)app.result.points[i].pnl
                  : (double)s->equity[at];
    }
    if (a < low)
      low = a;
    if (b > high)
      high = b;
  }
  if (high <= low)
    high = low + 1;
  int previous = 0;
  for (int i = 0; i < n; ++i) {
    int at = (s->head - s->count + i + HFT_TRACE) % HFT_TRACE,
        px = x + 8 + (w - 16) * i / n;
    if (candles) {
      HftCandle *c = &s->candles[at];
      int yy = y + h - 16 - (int)((c->high - low) / (high - low) * (h - 32));
      int bottom = y + h - 16 - (int)((c->low - low) / (high - low) * (h - 32));
      unsigned colorcode = c->close >= c->open ? CYAN : RED;
      line(px, yy, px, bottom, colorcode);
      int open = y + h - 16 - (int)((c->open - low) / (high - low) * (h - 32)),
          close =
              y + h - 16 - (int)((c->close - low) / (high - low) * (h - 32));
      rect(px - 2, open < close ? open : close, 4, abs(open - close) + 1,
           colorcode);
    } else {
      double val = !app.busy && app.result.point_count
                       ? (double)app.result.points[i].pnl
                       : (double)s->equity[at];
      int py = y + h - 16 - (int)((val - low) / (high - low) * (h - 32));
      if (i)
        line(x + 8 + (w - 16) * (i - 1) / n, previous, px, py, CYAN);
      previous = py;
    }
  }
  label(x + 12, y + 8, MUTED, "high %.2f", high);
  label(x + 12, y + h - 24, MUTED, "low %.2f", low);
}
static const char *formulas[13] = {
    "SMA: sign(mean_fast - mean_slow). Rolling integer sums.",
    "EMA: alpha = 2/(window+1); Q16 fixed-point recurrence.",
    "Mean reversion: trade outside z_enter; exit inside z_exit.",
    "Breakout: prior rolling high/low plus top-of-book volume confirmation.",
    "RSI: smoothed gain/(gain+loss); buy <30, sell >70.",
    "MACD: fast EMA - slow EMA; compare against 9-period signal.",
    "Imbalance: (bid_qty - ask_qty)/(bid_qty + ask_qty).",
    "Market maker: inventory skew and variance-dependent quote width.",
    "Pairs: rolling OLS hedge ratio and residual z-score; paired ownership.",
    "Execution: TWAP target = parent_qty * elapsed / horizon.",
    "MLP: 32 features -> 64 ReLU -> 32 ReLU -> 3 softmax.",
    "Q-learning: 81 states, 9 width/skew actions; inventory penalty.",
    "Selector: 4 k-means regimes; UCB1 selects SMA, reversion or maker."};
static void draw(void) {
  app.nhits = 0;
  int w = app.w, h = app.h, x = 224, content = w - x - 24;
  char value[256];
  rect(0, 0, w, h, BG);
  rect(0, 0, 200, h, 0x0d1727);
  textat(20, 26, "HFT / HPC", CYAN, 1);
  textat(20, 61, "STUDIO  /  C17", MUTED, 0);
  const char *nav[] = {"Overview", "Market",     "Strategy lab",   "Engines",
                       "Results",  "Cloud & AI", "Learn / source", "Settings"};
  for (int i = 0; i < 8; ++i)
    button(12, 118 + i * 48, 176, nav[i], 100 + i, app.page == i);
  info(20, h - 92, "LOCAL SIMULATION");
  info(20, h - 67, "Deterministic seed");
  label(20, h - 42, CYAN, "%.0f", (double)app.config.seed);
  textat(x, 25, nav[app.page], FG_TEXT, 1);
  button(w - 350, 22, 90, app.busy ? "Running" : "Run", 1, app.busy);
  button(w - 250, 22, 100, hft_load(&app.pause) ? "Resume" : "Pause", 2, 0);
  button(w - 140, 22, 110, "Stop", 3, 0);
  line(x, 78, w - 24, 78, BORDER);
  if (app.page == 0) {
    textat(x, 112, "A MARKET. EIGHT COMPUTE PATHS. MEASURED RESULTS.", CYAN, 1);
    info(x, 154,
         "Synthetic order flow, bounded order books, risk checks and "
         "reproducible strategies.");
    snprintf(value, sizeof(value), "%d symbols", app.config.symbols);
    card(x, 206, content / 3 - 12, "WORKLOAD", value, FG_TEXT);
    card(x + content / 3, 206, content / 3 - 12, "STRATEGY",
         hft_strategy_names[app.config.strategy], CYAN);
    card(x + 2 * content / 3, 206, content / 3 - 12, "ENGINE",
         hft_engine_names[app.config.engine], GOLD);
    plot(x, 326, content, 260, 1);
    button(x, 612, 200, "Run S1-S8 suite", 14, 0);
    button(x + 216, 612, 180, "Presentation mode", 41, app.presentation);
    info(x, 670,
         "1  Explore order flow     2  Choose a strategy     3  Compare "
         "engines     4  Export evidence");
  }
  if (app.page == 1) {
    snprintf(value, sizeof(value), "%.2f",
             app.snapshot.top.last * app.config.tick_size);
    card(x, 100, content / 4 - 10, "LAST PRICE", value, FG_TEXT);
    snprintf(value, sizeof(value), "%llu",
             (unsigned long long)app.snapshot.events);
    card(x + content / 4, 100, content / 4 - 10, "EVENTS / SYMBOL", value,
         CYAN);
    snprintf(value, sizeof(value), "%lld ticks", (long long)app.snapshot.pnl);
    card(x + content / 2, 100, content / 4 - 10, "SYMBOL P&L", value,
         app.snapshot.pnl >= 0 ? CYAN : RED);
    card(x + 3 * content / 4, 100, content / 4 - 10, "REGIME",
         hft_regime_names[app.snapshot.regime], GOLD);
    plot(x, 212, content * 2 / 3 - 16, 300, 1);
    textat(x + content * 2 / 3, 210, "DEPTH  /  PRICE-TIME FIFO", FG_TEXT, 0);
    for (int i = 0; i < 10; ++i) {
      int y = 245 + i * 24;
      int bw = app.snapshot.bid_qty[i] * 2;
      if (bw > 140)
        bw = 140;
      rect(x + content * 2 / 3, y, bw, 18, 0x17594f);
      snprintf(value, sizeof(value), "%lld  %d | %lld  %d",
               (long long)app.snapshot.bids[i], app.snapshot.bid_qty[i],
               (long long)app.snapshot.asks[i], app.snapshot.ask_qty[i]);
      textat(x + content * 2 / 3 + 4, y, value, FG_TEXT, 0);
    }
    textat(x, 540, "SYMBOL HEATMAP  /  RETURN FROM INITIAL PRICE", MUTED, 0);
    for (int i = 0; i < app.config.symbols && i < 64; ++i) {
      int xx = x + (i % 16) * (content / 16), yy = 572 + (i / 16) * 32;
      rect(xx, yy, content / 16 - 4, 28,
           app.snapshot.prices[i] >=
                   app.config.initial_price * app.config.tick_size
               ? 0x205d56
               : 0x613647);
      snprintf(value, sizeof(value), "S%02d", i);
      textat(xx + 8, yy + 5, value, FG_TEXT, 0);
    }
    snprintf(value, sizeof(value),
             "LAST STRATEGY FILL  %s %d @ %lld | inventory %lld",
             app.snapshot.last_fill.side ? "SELL" : "BUY",
             app.snapshot.last_fill.qty,
             (long long)app.snapshot.last_fill.price,
             (long long)(app.snapshot.count
                             ? app.snapshot
                                   .inventory[(app.snapshot.head + 255) % 256]
                             : 0));
    info(x, 728, value);
    button(x, 766, 180, "Cycle regime", 24, 0);
    button(x + 196, 766, 190, "Automatic regimes", 25, app.config.auto_regime);
  }
  if (app.page == 2) {
    for (int i = 0; i < 13; ++i)
      button(x + (i % 3) * (content / 3), 108 + (i / 3) * 45, content / 3 - 12,
             hft_strategy_names[i], 200 + i, app.config.strategy == i);
    textat(x, 370, formulas[app.config.strategy], CYAN, 0);
    info(x, 406,
         "All strategy orders pass the same position, exposure, collar, rate "
         "and loss checks.");
    snprintf(value, sizeof(value), "Fast window: %d", app.config.fast);
    button(x, 456, 230, value, 30, 0);
    snprintf(value, sizeof(value), "Slow window: %d", app.config.slow);
    button(x + 246, 456, 230, value, 31, 0);
    snprintf(value, sizeof(value), "Quantity: %d", app.config.quantity);
    button(x, 506, 230, value, 32, 0);
    snprintf(value, sizeof(value), "Position cap: %d", app.config.max_position);
    button(x + 246, 506, 230, value, 33, 0);
    info(x, 572,
         "More parameters are editable in configs/*.json; Load config is on "
         "Settings.");
    info(x, 608,
         "Pairs own two symbols together. Gross/loss budgets are divided "
         "equally by symbol.");
    info(x, 644, "AI model training and model loading are on Cloud & AI.");
  }
  if (app.page == 3) {
    for (int i = 0; i < 8; ++i) {
      HftCapability cap;
      hft_probe(i, &cap);
      button(x, 105 + i * 56, 190, hft_engine_names[i], 300 + i,
             app.config.engine == i);
      snprintf(value, sizeof(value), "%.90s", cap.reason);
      textat(x + 208, 114 + i * 56, value, cap.available ? MUTED : RED, 0);
    }
    snprintf(value, sizeof(value), "Symbols: %d", app.config.symbols);
    button(x, 582, 200, value, 20, 0);
    snprintf(value, sizeof(value), "Events: %llu",
             (unsigned long long)app.config.events);
    button(x + 216, 582, 200, value, 21, 0);
    snprintf(value, sizeof(value), "Workers: %d", app.config.threads);
    button(x, 630, 200, value, 22, 0);
    snprintf(value, sizeof(value), "Batch: %d", app.config.batch);
    button(x + 216, 630, 200, value, 23, 0);
    const char *schedules[] = {"Schedule: static", "Schedule: dynamic",
                               "Schedule: guided"};
    button(x, 678, 200, schedules[app.config.schedule], 35, 0);
    button(x + 216, 678, 200, "Run benchmark suite", 14, 0);
    info(x, 738,
         "cuBLAS/cuRAND: risk service. Custom device kernels use OpenCL C.");
  }
  if (app.page == 4) {
    HftResult *r = &app.result;
    snprintf(value, sizeof(value), "%.0f /s", r->throughput);
    card(x, 108, content / 3 - 12, "THROUGHPUT", value, CYAN);
    snprintf(value, sizeof(value), "%lld ticks", (long long)r->pnl);
    card(x + content / 3, 108, content / 3 - 12, "PORTFOLIO P&L", value,
         FG_TEXT);
    snprintf(value, sizeof(value), "%.2f ms",
             hft_hist_percentile(&r->latency, .99) / 1e6);
    card(x + 2 * content / 3, 108, content / 3 - 12, "CPU PAIR-STEP P99", value,
         GOLD);
    plot(x, 226, content, 300, 0);
    snprintf(value, sizeof(value),
             "Drawdown %.3f%% | Sharpe/sample %.4f | orders %llu | fills %llu",
             r->max_drawdown * 100, r->sharpe, (unsigned long long)r->orders,
             (unsigned long long)r->fills);
    info(x, 550, value);
    snprintf(value, sizeof(value),
             "H2D %.4fs | kernels %.4fs | D2H %.4fs | communication %.4fs",
             r->h2d_seconds, r->kernel_seconds, r->d2h_seconds,
             r->communication_seconds);
    info(x, 588, value);
    button(x, 644, 210, "Export report pack", 10, 0);
    button(x + 226, 644, 210, "Import desktop JSON", 11, 0);
    info(x, 706,
         "Reports include actual samples and hardware details; no fabricated "
         "backend scores.");
  }
  if (app.page == 5) {
    textat(x, 111, "CLOUD LATENCY LAB", CYAN, 1);
    info(x, 156,
         "Network delays are simulated; compute time is measured on this "
         "machine.");
    const char *networks[] = {"Local: 0 ns", "Colocated: 50 us",
                              "Same region: 500 us", "Cross-region: 20 ms"};
    button(x, 208, 250, networks[app.network], 27, 0);
    button(x + 266, 208, 240,
           app.config.mode ? "Latency-coupled mode" : "Verification mode", 26,
           app.config.mode);
    snprintf(value, sizeof(value), "Amplification: %.0fx",
             app.config.latency_amplification);
    button(x, 260, 250, value, 28, 0);
    snprintf(value, sizeof(value), "CPU price: $%.2f / hour",
             app.config.cpu_hourly);
    button(x + 266, 260, 240, value, 29, 0);
    label(x, 316, FG_TEXT, "Illustrative last-run cost: $%.8f",
          app.result.seconds * app.config.cpu_hourly / 3600);
    button(x, 364, 250, "Run latency experiment", 37, 0);
    textat(x, 438, "AI TRAINING  /  TEMPORAL VALIDATION", CYAN, 1);
    info(x, 480,
         "MLP uses a chronological split and a 16-event gap. Validation data "
         "stays held out.");
    button(x, 524, 210, "Train MLP", 12, app.config.model.trained);
    button(x + 226, 524, 210, "Train Q-learning", 13,
           app.config.model.q_trained);
    button(x, 578, 210, "Load model", 42, 0);
    label(x, 636, FG_TEXT, "Validation accuracy: %.2f%%", app.accuracy * 100);
    info(x, 686,
         "Example prices are editable assumptions, not current cloud-provider "
         "quotes.");
  }
  if (app.page == 6) {
    const char *titles[] = {
        "Unit 1: measurement",    "Unit 1: deterministic books",
        "Unit 2: CPU ownership",  "Unit 2: MPI collectives",
        "Unit 3: device memory",  "Unit 3: launch overhead",
        "Unit 4: model training", "Unit 4: network and cost"};
    const char *details[] = {"Warm up first. Repeat measurements. Compare "
                             "medians and inspect tail latency.",
                             "Same seed and indexed Philox counters produce "
                             "identical exogenous streams.",
                             "Each worker owns a symbol pair. Queue "
                             "publication uses acquire/release ordering.",
                             "Scatter pair IDs, broadcast config, reduce "
                             "metrics, gather final symbol states.",
                             "Host and device sizes are checked. Transfer and "
                             "kernel durations are separate.",
                             "Small batches pay fixed launch/transfer costs. "
                             "Throughput is not tick latency.",
                             "Train on past data. Normalize online. Validate "
                             "later; never tune using future labels.",
                             "Network presets add simulated delay. Cost = "
                             "measured hours * assumed price."};
    textat(x, 112, titles[app.lesson], CYAN, 1);
    info(x, 160, details[app.lesson]);
    button(x, 208, 180, "Previous lesson", 39, 0);
    button(x + 196, 208, 180, "Next lesson", 38, 0);
    textat(x, 282, "ACTUAL SHARED C SOURCE", FG_TEXT, 0);
    const char *p = hft_kernel_source;
    int skip = app.lesson * 35;
    for (int i = 0; i < skip && *p; ++i) {
      const char *end = strchr(p, '\n');
      p = end ? end + 1 : p + strlen(p);
    }
    for (int i = 0; i < 18 && *p; ++i) {
      const char *end = strchr(p, '\n');
      size_t len = end ? (size_t)(end - p) : strlen(p);
      if (len > 100)
        len = 100;
      char row[104];
      memcpy(row, p, len);
      row[len] = 0;
      textat(x, 324 + i * 21, row, i % 4 == 0 ? CYAN : MUTED, 0);
      p = end ? end + 1 : p + strlen(p);
    }
    button(x, 734, 260, "Export complete shared source", 40, 0);
  }
  if (app.page == 7) {
    button(x, 118, 230, "Load results/config.json", 43, 0);
    button(x, 172, 230, "Next deterministic seed", 36, 0);
    snprintf(value, sizeof(value), "Volatility: %.4f", app.config.volatility);
    button(x, 226, 230, value, 34, 0);
    button(x, 280, 230, "Reset current results", 4, 0);
    button(x, 334, 230, "Save PNG screenshot", 5, 0);
    info(x, 420,
         "Keyboard: Space run/pause, Escape stop, 1-8 navigate, F11 "
         "presentation, P screenshot.");
    info(x, 466,
         "Configuration controls are locked while a background task is "
         "running.");
    info(x, 512, "Native C desktop. No C++ UI library or C++ build step.");
    info(x, 558,
         "Scientific metric definitions and limitations are documented in "
         "docs/.");
  }
  rect(x, h - 64, content, 4, BORDER);
  rect(x, h - 64, (int)(content * app.snapshot.progress), 4, CYAN);
  snprintf(value, sizeof(value), "%.130s",
           app.busy ? app.snapshot.status : app.message);
  textat(x, h - 45, value, app.busy ? CYAN : MUTED, 0);
}
static void tick(void) {
  if (hft_snapshot_consume(&app.triple, &app.snapshot)) {
    if (app.snapshot.last_fill.qty)
      app.tape[app.tape_count++ % 32] = app.snapshot.last_fill;
  }
  if (app.busy && hft_load(&app.done)) {
    hft_thread_join(app.thread);
    app.busy = 0;
    if (app.job == 1) {
      hft_result_free(&app.result);
      app.result = app.working;
      memset(&app.working, 0, sizeof(app.working));
    }
    if (app.model_ready) {
      app.config.model = app.trained_model;
      app.loss = app.pending_loss;
      app.accuracy = app.pending_accuracy;
      app.reward = app.pending_reward;
    }

    if (app.result.states)
      hft_make_snapshot(&app.result, 0, &app.snapshot);
  }
  if (app.presentation && !app.busy && hft_now() - app.last_update > 8) {
    app.page = (app.page + 1) % 8;
    app.last_update = hft_now();
  }
}
static void key(int k) {
  if (k >= '1' && k <= '8')
    app.page = k - '1';
  if (k == ' ')
    click(app.busy ? 2 : 1);
  if (k == 27)
    click(3);
  if (k == 'p' || k == 'P')
    click(5);
}
static void mouse(int x, int y) {
  for (int i = 0; i < app.nhits; ++i) {
    Hit h = app.hits[i];
    if (x >= h.x && x < h.x + h.w && y >= h.y && y < h.y + h.h) {
      click(h.id);
      break;
    }
  }
}
static void init(void) {
  memset(&app, 0, sizeof(app));
  hft_config_default(&app.config);
  app.config.symbols = 16;
  app.config.events = 10000;
  app.w = 1440;
  app.h = 900;
  app.triple.back = 2;
  app.triple.front = 0;
  hft_store(&app.triple.middle, 1);
  strcpy(app.message, "Ready. Choose an engine and press Run.");
  app.last_update = hft_now();
}
static void shutdown_app(void) {
  if (app.busy) {
    hft_store(&app.stop, 1);
    hft_store(&app.pause, 0);
    hft_thread_join(app.thread);
  }
  hft_result_free(&app.result);
}
#ifdef _WIN32
static LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wp,
                                    LPARAM lp) {
  switch (message) {
  case WM_SIZE:
    app.w = LOWORD(lp);
    app.h = HIWORD(lp);
    return 0;
  case WM_TIMER:
    tick();
    InvalidateRect(hwnd, NULL, FALSE);
    return 0;
  case WM_ERASEBKGND:
    return 1;
  case WM_LBUTTONDOWN:
    mouse((short)LOWORD(lp), (short)HIWORD(lp));
    return 0;
  case WM_KEYDOWN:
    if (wp == VK_F11)
      click(41);
    else
      key((int)wp);
    return 0;
  case WM_PAINT: {
    PAINTSTRUCT ps;
    HDC screen = BeginPaint(hwnd, &ps);
    dc = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, app.w, app.h);
    HGDIOBJ old = SelectObject(dc, bitmap);
    draw();
    BitBlt(screen, 0, 0, app.w, app.h, dc, 0, 0, SRCCOPY);
    SelectObject(dc, old);
    DeleteObject(bitmap);
    DeleteDC(dc);
    EndPaint(hwnd, &ps);
    return 0;
  }
  case WM_DESTROY:
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProcA(hwnd, message, wp, lp);
}
int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR args,
                   int show) {
  (void)previous;
  (void)args;
  init();
  SetProcessDPIAware();
  WNDCLASSA wc = {0};
  wc.lpfnWndProc = window_proc;
  wc.hInstance = instance;
  wc.lpszClassName = "HftCStudio";
  wc.hCursor = LoadCursor(NULL, IDC_ARROW);
  RegisterClassA(&wc);
  font = CreateFontA(-16, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET,
                     OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                     DEFAULT_PITCH, "Segoe UI");
  heading = CreateFontA(-24, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET,
                        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
  window = CreateWindowExA(0, wc.lpszClassName, "HFT / HPC Studio - C",
                           WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                           1456, 940, NULL, NULL, instance, NULL);
  ShowWindow(window, show);
  SetTimer(window, 1, 33, NULL);
  MSG msg;
  while (GetMessageA(&msg, NULL, 0, 0) > 0) {
    TranslateMessage(&msg);
    DispatchMessageA(&msg);
  }
  shutdown_app();
  DeleteObject(font);
  DeleteObject(heading);
  return 0;
}
#else
int main(int argc, char **argv) {
  init();
  if (argc > 1 && !strcmp(argv[1], "--render")) {
    headless = 1;
    pixels = calloc((size_t)app.w * app.h, 3);
    if (!pixels)
      return 1;
    app.config.events = 2000;
    app.config.strategy = HFT_MAKER;
    if (!hft_run(&app.config, &app.result, NULL, NULL, NULL))
      return 1;
    hft_make_snapshot(&app.result, 0, &app.snapshot);
    hft_mkdir("preview");
    for (int page = 0; page < 8; ++page) {
      char path[128];
      app.page = page;
      draw();
      snprintf(path, sizeof(path), "preview/screen-%d.png", page + 1);
      if (!hft_write_png(path, pixels, app.w, app.h, app.message))
        return 1;
    }
    free(pixels);
    shutdown_app();
    return 0;
  }

  int smoke = argc > 1 && !strcmp(argv[1], "--smoke");
  display = XOpenDisplay(NULL);
  if (!display) {
    fprintf(stderr, "No X display. Use hft_bench on headless systems.\n");
    return 1;
  }
  int screen = DefaultScreen(display);
  window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0,
                               (unsigned)app.w, (unsigned)app.h, 0, 0, BG);
  XStoreName(display, window, "HFT / HPC Studio - C");
  XSelectInput(display, window,
               ExposureMask | KeyPressMask | ButtonPressMask |
                   StructureNotifyMask);
  Atom close = XInternAtom(display, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(display, window, &close, 1);
  gc = XCreateGC(display, window, 0, NULL);
  font = XLoadQueryFont(display, "fixed");
  if (font)
    XSetFont(display, gc, font->fid);
  canvas = XCreatePixmap(display, window, (unsigned)app.w, (unsigned)app.h,
                         (unsigned)DefaultDepth(display, screen));
  XMapWindow(display, window);
  int quit = 0;
  double begin = hft_now();
  if (smoke) {
    app.page = 1;
    start(1);
  }
  while (!quit) {
    while (XPending(display)) {
      XEvent ev;
      XNextEvent(display, &ev);
      if (ev.type == ClientMessage)
        quit = 1;
      else if (ev.type == ButtonPress)
        mouse(ev.xbutton.x, ev.xbutton.y);
      else if (ev.type == KeyPress) {
        KeySym sym = XLookupKeysym(&ev.xkey, 0);
        if (sym == XK_Escape)
          key(27);
        else if (sym == XK_F11)
          click(41);
        else
          key((int)sym);
      } else if (ev.type == ConfigureNotify) {
        if (ev.xconfigure.width != app.w || ev.xconfigure.height != app.h) {
          app.w = ev.xconfigure.width;
          app.h = ev.xconfigure.height;
          XFreePixmap(display, canvas);
          canvas =
              XCreatePixmap(display, window, (unsigned)app.w, (unsigned)app.h,
                            (unsigned)DefaultDepth(display, screen));
        }
      }
    }
    tick();
    draw();
    XCopyArea(display, canvas, window, gc, 0, 0, (unsigned)app.w,
              (unsigned)app.h, 0, 0);
    XFlush(display);
    if (smoke && hft_now() - begin > 3 && !app.busy) {
      save_screenshot();
      quit = 1;
    }
    hft_sleep(33);
  }
  shutdown_app();
  XFreePixmap(display, canvas);
  XFreeGC(display, gc);
  if (font)
    XFreeFont(display, font);
  XDestroyWindow(display, window);
  XCloseDisplay(display);
  return 0;
}
#endif
