/* Purpose: C OS abstraction. Unit 1: clocks, threads, atomics and affinity.
 * No C++ runtime; O(1) queue operations with acquire/release ownership. */
#ifndef HFT_PLATFORM_H
#define HFT_PLATFORM_H
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef struct {
  volatile LONG64 value;
} HftAtomic;
typedef HANDLE HftThread;
typedef CRITICAL_SECTION HftMutex;
#else
#include <pthread.h>
#include <stdatomic.h>
typedef struct {
  _Atomic uint64_t value;
} HftAtomic;
typedef pthread_t HftThread;
typedef pthread_mutex_t HftMutex;
#endif
uint64_t hft_load(const HftAtomic *a);
void hft_store(HftAtomic *a, uint64_t value);
uint64_t hft_exchange(HftAtomic *a, uint64_t value);
uint64_t hft_add(HftAtomic *a, uint64_t value);
typedef void *(*HftThreadFn)(void *);
int hft_thread_start(HftThread *, HftThreadFn, void *);
void hft_thread_join(HftThread);
void hft_mutex_init(HftMutex *);
void hft_mutex_lock(HftMutex *);
void hft_mutex_unlock(HftMutex *);
void hft_mutex_destroy(HftMutex *);
double hft_now(void);
void hft_sleep(unsigned milliseconds);
int hft_cores(void);
int hft_pin_thread(int core);
void hft_system_info(char *text, size_t capacity);
int hft_mkdir(const char *path);
int hft_file_exists(const char *path);
typedef struct {
  HftAtomic head;
  char pad1[64];
  HftAtomic tail;
  char pad2[64];
  int items[64];
} HftQueue;
int hft_queue_push(HftQueue *, int);
int hft_queue_pop(HftQueue *, int *);
#endif
