/* Purpose: actual OS facilities in C. Unit 1: threads, affinity and monotonic
 * time. Thread starts allocate outside measured hot loops. Queue
 * synchronization is single-producer/single-consumer acquire/release; O(1)
 * operations. */
#define _GNU_SOURCE
#include "hft/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#include <process.h>
uint64_t hft_load(const HftAtomic *a) {
  return (uint64_t)InterlockedCompareExchange64((volatile LONG64 *)&a->value, 0,
                                                0);
}
void hft_store(HftAtomic *a, uint64_t v) {
  InterlockedExchange64(&a->value, (LONG64)v);
}
uint64_t hft_exchange(HftAtomic *a, uint64_t v) {
  return (uint64_t)InterlockedExchange64(&a->value, (LONG64)v);
}
uint64_t hft_add(HftAtomic *a, uint64_t v) {
  return (uint64_t)InterlockedExchangeAdd64(&a->value, (LONG64)v);
}
typedef struct {
  HftThreadFn fn;
  void *arg;
} Start;
static unsigned __stdcall trampoline(void *p) {
  Start s = *(Start *)p;
  free(p);
  s.fn(s.arg);
  return 0;
}
int hft_thread_start(HftThread *t, HftThreadFn fn, void *arg) {
  Start *s = (Start *)malloc(sizeof(*s));
  if (!s)
    return 0;
  s->fn = fn;
  s->arg = arg;
  *t = (HANDLE)_beginthreadex(NULL, 0, trampoline, s, 0, NULL);
  if (!*t) {
    free(s);
    return 0;
  }
  return 1;
}
void hft_thread_join(HftThread t) {
  WaitForSingleObject(t, INFINITE);
  CloseHandle(t);
}
void hft_mutex_init(HftMutex *m) { InitializeCriticalSection(m); }
void hft_mutex_lock(HftMutex *m) { EnterCriticalSection(m); }
void hft_mutex_unlock(HftMutex *m) { LeaveCriticalSection(m); }
void hft_mutex_destroy(HftMutex *m) { DeleteCriticalSection(m); }
double hft_now(void) {
  LARGE_INTEGER t, f;
  QueryPerformanceCounter(&t);
  QueryPerformanceFrequency(&f);
  return (double)t.QuadPart / (double)f.QuadPart;
}
void hft_sleep(unsigned ms) { Sleep(ms); }
int hft_cores(void) {
  SYSTEM_INFO s;
  GetSystemInfo(&s);
  return (int)s.dwNumberOfProcessors;
}
int hft_pin_thread(int core) {
  return core >= 0 && core < 64 &&
         SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core) != 0;
}
int hft_mkdir(const char *p) {
  return _mkdir(p) == 0 || GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES;
}
void hft_system_info(char *text, size_t n) {
  MEMORYSTATUSEX m;
  char cpu[256] = "Windows CPU";
  DWORD size = sizeof(cpu);
  HKEY key;
  memset(&m, 0, sizeof(m));
  m.dwLength = sizeof(m);
  GlobalMemoryStatusEx(&m);
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                    "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                    KEY_READ, &key) == ERROR_SUCCESS) {
    RegQueryValueExA(key, "ProcessorNameString", NULL, NULL, (LPBYTE)cpu,
                     &size);
    RegCloseKey(key);
  }
  snprintf(text, n, "%s | logical CPUs: %d | RAM: %.1f GiB | Windows x64", cpu,
           hft_cores(), (double)m.ullTotalPhys / 1073741824.0);
}
#else
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>
uint64_t hft_load(const HftAtomic *a) {
  return atomic_load_explicit(&a->value, memory_order_acquire);
}
void hft_store(HftAtomic *a, uint64_t v) {
  atomic_store_explicit(&a->value, v, memory_order_release);
}
uint64_t hft_exchange(HftAtomic *a, uint64_t v) {
  return atomic_exchange_explicit(&a->value, v, memory_order_acq_rel);
}
uint64_t hft_add(HftAtomic *a, uint64_t v) {
  return atomic_fetch_add_explicit(&a->value, v, memory_order_acq_rel);
}
int hft_thread_start(HftThread *t, HftThreadFn fn, void *arg) {
  return pthread_create(t, NULL, fn, arg) == 0;
}
void hft_thread_join(HftThread t) { pthread_join(t, NULL); }
void hft_mutex_init(HftMutex *m) { pthread_mutex_init(m, NULL); }
void hft_mutex_lock(HftMutex *m) { pthread_mutex_lock(m); }
void hft_mutex_unlock(HftMutex *m) { pthread_mutex_unlock(m); }
void hft_mutex_destroy(HftMutex *m) { pthread_mutex_destroy(m); }
double hft_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
void hft_sleep(unsigned ms) {
  struct timespec t = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
  nanosleep(&t, NULL);
}
int hft_cores(void) {
  cpu_set_t mask;
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    return CPU_COUNT(&mask);
  return (int)sysconf(_SC_NPROCESSORS_ONLN);
}
int hft_pin_thread(int core) {
  cpu_set_t allowed, chosen;
  int i, n = 0;
  if (sched_getaffinity(0, sizeof(allowed), &allowed))
    return 0;
  CPU_ZERO(&chosen);
  for (i = 0; i < CPU_SETSIZE; ++i)
    if (CPU_ISSET(i, &allowed)) {
      if (n++ == core) {
        CPU_SET(i, &chosen);
        return pthread_setaffinity_np(pthread_self(), sizeof(chosen),
                                      &chosen) == 0;
      }
    }
  return 0;
}
int hft_mkdir(const char *p) {
  struct stat st;
  return mkdir(p, 0755) == 0 || (stat(p, &st) == 0 && S_ISDIR(st.st_mode));
}
void hft_system_info(char *text, size_t n) {
  FILE *f = fopen("/proc/cpuinfo", "r");
  char line[512], cpu[256] = "Unknown CPU";
  struct utsname u;
  long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGESIZE);
  if (f) {
    while (fgets(line, sizeof(line), f))
      if (strncmp(line, "model name", 10) == 0) {
        char *p = strchr(line, ':');
        if (p) {
          snprintf(cpu, sizeof(cpu), "%.240s", p + 2);
          cpu[strcspn(cpu, "\r\n")] = 0;
        }
        break;
      }
    fclose(f);
  }
  memset(&u, 0, sizeof(u));
  uname(&u);
  snprintf(text, n, "%s | available CPUs: %d | RAM: %.1f GiB | %s %s", cpu,
           hft_cores(), (double)pages * (double)size / 1073741824.0, u.sysname,
           u.release);
}
#endif
int hft_file_exists(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f)
    return 0;
  fclose(f);
  return 1;
}
int hft_queue_push(HftQueue *q, int item) {
  uint64_t h = hft_load(&q->head), next = (h + 1) & 63;
  if (next == hft_load(&q->tail))
    return 0;
  q->items[h] = item;
  hft_store(&q->head, next);
  return 1;
}
int hft_queue_pop(HftQueue *q, int *item) {
  uint64_t t = hft_load(&q->tail);
  if (t == hft_load(&q->head))
    return 0;
  *item = q->items[t];
  hft_store(&q->tail, (t + 1) & 63);
  return 1;
}
