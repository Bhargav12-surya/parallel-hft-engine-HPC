/* C host interface to OpenCL C, with measured transfers, device profiling,
 * exact-layout check and explicit capability failures. No emulated GPU timings.
 */
#include "hft/api.h"
#include "hft/numeric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HFT_HAVE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include "kernels.inc"
#include <CL/cl.h>
typedef struct {
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  char name[256];
} Device;
static int device_open(Device *d, int compile, char *error) {
  cl_uint np = 0;
  cl_int rc;
  cl_platform_id platforms[16];
  memset(d, 0, sizeof(*d));
  rc = clGetPlatformIDs(16, platforms, &np);
  if (rc != CL_SUCCESS || !np) {
    strcpy(error, "No OpenCL platform/driver installed");
    return 0;
  }
  cl_device_type type =
      getenv("HFT_OPENCL_CPU") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;
  for (cl_uint p = 0; p < np && p < 16 && !d->device; ++p) {
    cl_device_id dev;
    cl_device_fp_config fp = 0;
    if (clGetDeviceIDs(platforms[p], type, 1, &dev, NULL) == CL_SUCCESS) {
      clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp), &fp, NULL);
      if (fp)
        d->device = dev;
    }
  }
  if (!d->device) {
    strcpy(error, "No fp64 OpenCL GPU; HFT_OPENCL_CPU=1 explicitly selects a "
                  "CPU validation device");
    return 0;
  }
  clGetDeviceInfo(d->device, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);
  if (!compile)
    return 1;
  d->context = clCreateContext(NULL, 1, &d->device, NULL, NULL, &rc);
  if (rc != CL_SUCCESS)
    goto fail;
  d->queue = clCreateCommandQueue(d->context, d->device,
                                  CL_QUEUE_PROFILING_ENABLE, &rc);
  if (rc != CL_SUCCESS)
    goto fail;
  const char *source = hft_kernel_source;
  d->program = clCreateProgramWithSource(d->context, 1, &source, NULL, &rc);
  if (rc != CL_SUCCESS)
    goto fail;
  rc = clBuildProgram(d->program, 1, &d->device, "-cl-std=CL1.2", NULL, NULL);
  if (rc != CL_SUCCESS) {
    clGetProgramBuildInfo(d->program, d->device, CL_PROGRAM_BUILD_LOG,
                          HFT_ERROR - 1, error, NULL);
    error[HFT_ERROR - 1] = 0;
    return 0;
  }
  return 1;
fail:
  snprintf(error, HFT_ERROR, "OpenCL initialization failed (%d)", rc);
  return 0;
}
static void device_close(Device *d) {
  if (d->program)
    clReleaseProgram(d->program);
  if (d->queue)
    clReleaseCommandQueue(d->queue);
  if (d->context)
    clReleaseContext(d->context);
}
static double elapsed(cl_event ev) {
  cl_ulong a = 0, b = 0;
  clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(a), &a, NULL);
  clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(b), &b, NULL);
  clReleaseEvent(ev);
  return (b - a) * 1e-9;
}
static int layout_ok(Device *d, char *error) {
  cl_int rc;
  cl_ulong sizes[3] = {0};
  cl_mem b =
      clCreateBuffer(d->context, CL_MEM_WRITE_ONLY, sizeof(sizes), NULL, &rc);
  cl_kernel k = clCreateKernel(d->program, "layout", &rc);
  size_t n = 1;
  if (!b || !k) {
    if (b)
      clReleaseMemObject(b);
    if (k)
      clReleaseKernel(k);
    return 0;
  }
  clSetKernelArg(k, 0, sizeof(b), &b);
  rc = clEnqueueNDRangeKernel(d->queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
  rc |= clEnqueueReadBuffer(d->queue, b, CL_TRUE, 0, sizeof(sizes), sizes, 0,
                            NULL, NULL);
  clReleaseKernel(k);
  clReleaseMemObject(b);
  if (rc || sizes[0] != sizeof(HftSymbol) || sizes[1] != sizeof(HftConfig) ||
      sizes[2] != sizeof(HftModel)) {
    strcpy(error, "Host/device structure layout mismatch");
    return 0;
  }
  return 1;
}
void hft_opencl_probe(HftCapability *c) {
  Device d;
  char e[HFT_ERROR] = {0};
  memset(c, 0, sizeof(*c));
  c->available = device_open(&d, 0, e);
  snprintf(c->name, sizeof(c->name), "%.127s",
           c->available ? d.name : "OpenCL C");
  snprintf(c->reason, sizeof(c->reason), "%.255s",
           c->available
               ? (getenv("HFT_OPENCL_CPU") ? "CPU OpenCL validation device"
                                           : "GPU available")
               : e);
}
static int transfer_states(Device *d, cl_mem device, cl_mem staging, void *host,
                           size_t bytes, int to_device, double *duration) {
  cl_int rc;
  cl_event ev;
  void *mapped;
  if (!staging) {
    if (to_device)
      rc = clEnqueueWriteBuffer(d->queue, device, CL_TRUE, 0, bytes, host, 0,
                                NULL, &ev);
    else
      rc = clEnqueueReadBuffer(d->queue, device, CL_TRUE, 0, bytes, host, 0,
                               NULL, &ev);
    if (rc)
      return 0;
    *duration += elapsed(ev);
    return 1;
  }
  if (to_device) {
    mapped = clEnqueueMapBuffer(d->queue, staging, CL_TRUE, CL_MAP_WRITE, 0,
                                bytes, 0, NULL, NULL, &rc);
    if (rc || !mapped)
      return 0;
    memcpy(mapped, host, bytes);
    rc = clEnqueueUnmapMemObject(d->queue, staging, mapped, 0, NULL, NULL);
    if (rc)
      return 0;
    rc = clEnqueueCopyBuffer(d->queue, staging, device, 0, 0, bytes, 0, NULL,
                             &ev);
  } else
    rc = clEnqueueCopyBuffer(d->queue, device, staging, 0, 0, bytes, 0, NULL,
                             &ev);
  if (rc)
    return 0;
  clWaitForEvents(1, &ev);
  *duration += elapsed(ev);
  if (!to_device) {
    mapped = clEnqueueMapBuffer(d->queue, staging, CL_TRUE, CL_MAP_READ, 0,
                                bytes, 0, NULL, NULL, &rc);
    if (rc || !mapped)
      return 0;
    memcpy(host, mapped, bytes);
    rc = clEnqueueUnmapMemObject(d->queue, staging, mapped, 0, NULL, NULL);
    if (rc)
      return 0;
    clFinish(d->queue);
  }
  return 1;
}
int hft_run_opencl(const HftConfig *c, HftResult *r, HftAtomic *cancel,
                   HftProgress progress, void *user) {
  Device d = {0};
  cl_int rc = 0;
  cl_mem states = NULL, cfg = NULL, staging = NULL;
  cl_kernel kernels[2] = {NULL, NULL};
  cl_command_queue queues[2] = {NULL, NULL};
  cl_event events[2];
  double start;
  int ok = 0, groups = (c->symbols + 1) / 2,
      nq = c->streams == 2 && groups > 1 ? 2 : 1;
  hft_u64 done = 0, batches = 0;
  size_t bytes = (size_t)c->symbols * sizeof(HftSymbol);
  if (!hft_initialize(c, r))
    return 0;
  if (!device_open(&d, 1, r->error) || !layout_ok(&d, r->error))
    goto end;
  snprintf(r->device, sizeof(r->device), "%s", d.name);
  cl_ulong maxalloc = 0;
  clGetDeviceInfo(d.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc),
                  &maxalloc, NULL);
  if (bytes > maxalloc) {
    strcpy(r->error, "Symbol state exceeds device maximum allocation");
    goto end;
  }
  states = clCreateBuffer(d.context, CL_MEM_READ_WRITE, bytes, NULL, &rc);
  cfg = clCreateBuffer(d.context, CL_MEM_READ_ONLY, sizeof(*c), NULL, &rc);
  if (c->pinned)
    staging = clCreateBuffer(
        d.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, NULL, &rc);
  if (!states || !cfg || (c->pinned && !staging))
    goto end;
  queues[0] = d.queue;
  if (nq == 2) {
    queues[1] = clCreateCommandQueue(d.context, d.device,
                                     CL_QUEUE_PROFILING_ENABLE, &rc);
    if (!queues[1])
      goto end;
  }
  size_t local = (size_t)c->block_size, global[2];
  for (int q = 0; q < nq; ++q) {
    kernels[q] = clCreateKernel(d.program, "simulate", &rc);
    if (!kernels[q])
      goto end;
    size_t limit;
    clGetKernelWorkGroupInfo(kernels[q], d.device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(limit), &limit, NULL);
    if (local > limit)
      local = limit;
    int first = q ? groups / 2 : 0, last = nq == 2 && !q ? groups / 2 : groups;
    global[q] = ((size_t)(last - first) + local - 1) / local * local;
    rc = clSetKernelArg(kernels[q], 0, sizeof(states), &states);
    rc |= clSetKernelArg(kernels[q], 1, sizeof(cfg), &cfg);
    rc |= clSetKernelArg(kernels[q], 2, sizeof(last), &last);
    rc |= clSetKernelArg(kernels[q], 3, sizeof(first), &first);
    if (rc)
      goto end;
  }
  start = hft_now();
  if (!transfer_states(&d, states, staging, r->states, bytes, 1,
                       &r->h2d_seconds))
    goto end;
  rc = clEnqueueWriteBuffer(d.queue, cfg, CL_TRUE, 0, sizeof(*c), c, 0, NULL,
                            &events[0]);
  if (rc)
    goto end;
  r->h2d_seconds += elapsed(events[0]);
  while (done < c->events && (!cancel || !hft_load(cancel))) {
    hft_u64 steps = c->events - done;
    if (steps > (hft_u64)c->batch)
      steps = c->batch;
    hft_u64 delay = 0;
    if (c->mode) {
      HftRandom rnd = hft_random(c->seed, 0, 501, batches);
      double ns = c->network_ns +
                  c->network_jitter_ns * hft_normal(rnd.x[0], rnd.x[1]) +
                  (done ? (hft_now() - start) * 1e9 / (done * c->symbols) *
                              c->latency_amplification
                        : 0);
      delay = (hft_u64)hft_max(0, ns);
    }
    int submitted = 0;
    for (int q = 0; q < nq; ++q) {
      rc = clSetKernelArg(kernels[q], 4, sizeof(steps), &steps);
      rc |= clSetKernelArg(kernels[q], 5, sizeof(delay), &delay);
      if (rc)
        break;
      rc = clEnqueueNDRangeKernel(queues[q], kernels[q], 1, NULL, &global[q],
                                  &local, 0, NULL, &events[q]);
      if (rc)
        break;
      clFlush(queues[q]);
      ++submitted;
    }
    for (int q = 0; q < submitted; ++q) {
      clWaitForEvents(1, &events[q]);
      r->kernel_seconds += elapsed(events[q]);
    }
    if (rc)
      goto end;
    done += steps;
    ++batches;
    if (progress || done == c->events || batches % 16 == 0) {
      if (!transfer_states(&d, states, staging, r->states, bytes, 0,
                           &r->d2h_seconds))
        goto end;
      hft_result_finalize(r);
      if (r->point_count < 2049) {
        HftPoint *p = r->points + r->point_count++;
        p->event = done * c->symbols;
        p->pnl = r->pnl;
      }
      if (progress) {
        HftSnapshot snap;
        hft_make_snapshot(r, 0, &snap);
        snap.progress = (double)done / c->events;
        snap.events_second = (double)(done * c->symbols) / (hft_now() - start);
        progress(&snap, user);
      }
    }
  }
  if (!transfer_states(&d, states, staging, r->states, bytes, 0,
                       &r->d2h_seconds))
    goto end;
  r->seconds = hft_now() - start;
  r->cancelled = done < c->events;
  r->batch_latency_ns = batches ? r->kernel_seconds * 1e9 / batches : 0;
  r->workers = groups;
  hft_result_finalize(r);
  ok = 1;
end:
  if (!ok && !r->error[0])
    snprintf(r->error, HFT_ERROR, "OpenCL execution/allocation failure (%d)",
             rc);
  for (int q = 0; q < 2; ++q)
    if (kernels[q])
      clReleaseKernel(kernels[q]);
  if (queues[1])
    clReleaseCommandQueue(queues[1]);
  if (staging)
    clReleaseMemObject(staging);
  if (states)
    clReleaseMemObject(states);
  if (cfg)
    clReleaseMemObject(cfg);
  device_close(&d);
  return ok;
}
static int compute(int kind, HftModel *model, const float *x, const int *labels,
                   int rows, int cols, float *out, int epochs, int tiled,
                   double *seconds, char *error) {
  Device d;
  cl_mem a = NULL, b = NULL, m = NULL, y = NULL;
  cl_kernel k = NULL, u = NULL;
  cl_int rc = 0;
  int ok = 0;
  if (!device_open(&d, 1, error))
    goto end;
  size_t asize = (size_t)rows * cols * sizeof(float),
         bsize =
             (size_t)(kind == 0 ? cols * cols
                                : (kind == 1 ? rows * 3 : 32 * HFT_WEIGHTS)) *
             sizeof(float);
  a = clCreateBuffer(d.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, asize,
                     (void *)x, &rc);
  b = clCreateBuffer(d.context, CL_MEM_READ_WRITE, bsize, NULL, &rc);
  if (kind)
    m = clCreateBuffer(d.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(*model), model, &rc);
  if (kind == 2)
    y = clCreateBuffer(d.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       (size_t)rows * sizeof(int), (void *)labels, &rc);
  k = clCreateKernel(d.program,
                     kind == 0 ? (tiled ? "covariance_tiled" : "covariance")
                               : (kind == 1 ? "inference" : "gradient"),
                     &rc);
  if (!a || !b || !k || (kind && !m) || (kind == 2 && !y))
    goto end;
  double start = hft_now();
  if (kind == 0) {
    clSetKernelArg(k, 0, sizeof(a), &a);
    clSetKernelArg(k, 1, sizeof(b), &b);
    clSetKernelArg(k, 2, sizeof(rows), &rows);
    clSetKernelArg(k, 3, sizeof(cols), &cols);
    clSetKernelArg(k, 4, sizeof(tiled), &tiled);
    size_t n[2] = {(size_t)cols, (size_t)cols}, local[2] = {8, 8};
    if (tiled)
      n[0] = n[1] = ((size_t)cols + 7) / 8 * 8;
    rc = clEnqueueNDRangeKernel(d.queue, k, 2, NULL, n, tiled ? local : NULL, 0,
                                NULL, NULL);
  } else if (kind == 1) {
    clSetKernelArg(k, 0, sizeof(m), &m);
    clSetKernelArg(k, 1, sizeof(a), &a);
    clSetKernelArg(k, 2, sizeof(b), &b);
    clSetKernelArg(k, 3, sizeof(rows), &rows);
    size_t n = (size_t)rows;
    rc = clEnqueueNDRangeKernel(d.queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
  } else {
    u = clCreateKernel(d.program, "update", &rc);
    if (!u)
      goto end;
    clSetKernelArg(k, 0, sizeof(m), &m);
    clSetKernelArg(k, 1, sizeof(a), &a);
    clSetKernelArg(k, 2, sizeof(y), &y);
    clSetKernelArg(k, 3, sizeof(b), &b);
    clSetKernelArg(u, 0, sizeof(m), &m);
    clSetKernelArg(u, 1, sizeof(b), &b);
    for (int ep = 0; ep < epochs; ++ep)
      for (int begin = 0; begin < rows; begin += 32) {
        int count = rows - begin;
        if (count > 32)
          count = 32;
        size_t n = (size_t)count, weights = HFT_WEIGHTS;
        clSetKernelArg(k, 4, sizeof(count), &count);
        clSetKernelArg(k, 5, sizeof(begin), &begin);
        clSetKernelArg(u, 2, sizeof(count), &count);
        rc |= clEnqueueNDRangeKernel(d.queue, k, 1, NULL, &n, NULL, 0, NULL,
                                     NULL);
        rc |= clEnqueueNDRangeKernel(d.queue, u, 1, NULL, &weights, NULL, 0,
                                     NULL, NULL);
      }
  }
  if (kind == 2) {
    rc |= clEnqueueReadBuffer(d.queue, m, CL_TRUE, 0, sizeof(*model), model, 0,
                              NULL, NULL);
    model->trained = 1;
  } else
    rc |=
        clEnqueueReadBuffer(d.queue, b, CL_TRUE, 0, bsize, out, 0, NULL, NULL);
  *seconds = hft_now() - start;
  ok = rc == CL_SUCCESS;
end:
  if (!ok && !error[0])
    snprintf(error, HFT_ERROR, "OpenCL analytics failed (%d)", rc);
  if (u)
    clReleaseKernel(u);
  if (k)
    clReleaseKernel(k);
  if (a)
    clReleaseMemObject(a);
  if (b)
    clReleaseMemObject(b);
  if (m)
    clReleaseMemObject(m);
  if (y)
    clReleaseMemObject(y);
  device_close(&d);
  return ok;
}
int hft_opencl_matrix(const float *x, int rows, int cols, float *out, int tiled,
                      double *seconds, char *error) {
  return compute(0, NULL, x, NULL, rows, cols, out, 0, tiled, seconds, error);
}
int hft_opencl_infer(const HftModel *m, const float *x, int rows, float *out,
                     double *seconds, char *error) {
  return compute(1, (HftModel *)m, x, NULL, rows, 32, out, 0, 0, seconds,
                 error);
}
int hft_opencl_train(HftModel *m, const float *x, const int *labels, int rows,
                     int epochs, double *seconds, char *error) {
  return compute(2, m, x, labels, rows, 32, NULL, epochs, 0, seconds, error);
}
int hft_opencl_qtrain(HftModel *m, int episodes, int steps, hft_u64 seed,
                      double *seconds, char *error) {
  Device d;
  cl_mem states = NULL, cfg = NULL;
  cl_kernel k = NULL;
  cl_int rc = 0;
  int ok = 0;
  HftSymbol *host = calloc((size_t)episodes * 2, sizeof(HftSymbol));
  HftConfig *configs = calloc((size_t)episodes, sizeof(HftConfig));
  if (!host || !configs) {
    free(host);
    free(configs);
    return 0;
  }
  if (!device_open(&d, 1, error))
    goto end;
  for (int i = 0; i < episodes; ++i) {
    HftResult r;
    hft_config_default(configs + i);
    configs[i].symbols = 2;
    configs[i].strategy = HFT_QLEARNING;
    configs[i].seed = seed + (hft_u64)i * 7919;
    configs[i].model = *m;
    if (!hft_initialize(configs + i, &r))
      goto end;
    memcpy(host + i * 2, r.states, 2 * sizeof(HftSymbol));
    hft_result_free(&r);
  }
  double start = hft_now();
  states = clCreateBuffer(d.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          (size_t)episodes * 2 * sizeof(HftSymbol), host, &rc);
  cfg = clCreateBuffer(d.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       (size_t)episodes * sizeof(HftConfig), configs, &rc);
  k = clCreateKernel(d.program, "qlearning", &rc);
  if (!states || !cfg || !k)
    goto end;
  clSetKernelArg(k, 0, sizeof(states), &states);
  clSetKernelArg(k, 1, sizeof(cfg), &cfg);
  clSetKernelArg(k, 2, sizeof(episodes), &episodes);
  clSetKernelArg(k, 3, sizeof(steps), &steps);
  size_t n = (size_t)episodes;
  rc = clEnqueueNDRangeKernel(d.queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
  rc |= clEnqueueReadBuffer(d.queue, cfg, CL_TRUE, 0,
                            (size_t)episodes * sizeof(HftConfig), configs, 0,
                            NULL, NULL);
  *seconds = hft_now() - start;
  if (rc)
    goto end;
  for (int j = 0; j < HFT_QSTATES * 9; ++j) {
    double sum = 0;
    for (int i = 0; i < episodes; ++i)
      sum += configs[i].model.q[j];
    m->q[j] = (float)(sum / episodes);
  }
  m->q_trained = 1;
  ok = 1;
end:
  if (k)
    clReleaseKernel(k);
  if (states)
    clReleaseMemObject(states);
  if (cfg)
    clReleaseMemObject(cfg);
  device_close(&d);
  free(host);
  free(configs);
  return ok;
}
#else
static int unavailable(char *e) {
  strcpy(e, "OpenCL headers/library were not available when this C build was "
            "configured");
  return 0;
}
void hft_opencl_probe(HftCapability *c) {
  memset(c, 0, sizeof(*c));
  strcpy(c->name, "OpenCL C");
  strcpy(c->reason,
         "OpenCL C support not compiled; install an OpenCL SDK and rebuild");
}
int hft_run_opencl(const HftConfig *c, HftResult *r, HftAtomic *stop,
                   HftProgress cb, void *u) {
  (void)c;
  (void)stop;
  (void)cb;
  (void)u;
  return unavailable(r->error);
}
int hft_opencl_matrix(const float *x, int r, int c, float *o, int t, double *s,
                      char *e) {
  (void)x;
  (void)r;
  (void)c;
  (void)o;
  (void)t;
  (void)s;
  return unavailable(e);
}
int hft_opencl_infer(const HftModel *m, const float *x, int r, float *o,
                     double *s, char *e) {
  (void)m;
  (void)x;
  (void)r;
  (void)o;
  (void)s;
  return unavailable(e);
}
int hft_opencl_train(HftModel *m, const float *x, const int *y, int r, int ep,
                     double *s, char *e) {
  (void)m;
  (void)x;
  (void)y;
  (void)r;
  (void)ep;
  (void)s;
  return unavailable(e);
}
int hft_opencl_qtrain(HftModel *m, int ep, int st, hft_u64 seed, double *s,
                      char *e) {
  (void)m;
  (void)ep;
  (void)st;
  (void)seed;
  (void)s;
  return unavailable(e);
}
#endif
