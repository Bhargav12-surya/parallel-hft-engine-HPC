#define _GNU_SOURCE
#include "hft_engine.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
double hft_now(void) {
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks); QueryPerformanceFrequency(&frequency);
    return (double)ticks.QuadPart / (double)frequency.QuadPart;
}
int hft_logical_cpus(void) { SYSTEM_INFO info; GetSystemInfo(&info); return (int)info.dwNumberOfProcessors; }
int hft_pin_current_thread(int single_core) {
    if (!single_core) return 0;
    DWORD_PTR eligible = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &eligible, &system) || !eligible) return 0;
    DWORD_PTR lowest = eligible & (~eligible + 1);
    return SetThreadAffinityMask(GetCurrentThread(), lowest) != 0;
}
int hft_atomic_replace(const char *source,const char *destination) {
    return MoveFileExA(source,destination,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH) != 0;
}
#else
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif
double hft_now(void) { struct timespec value; clock_gettime(CLOCK_MONOTONIC, &value); return value.tv_sec + value.tv_nsec * 1e-9; }
int hft_logical_cpus(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1; }
int hft_pin_current_thread(int single_core) {
#if defined(__linux__)
    if (!single_core) return 0;
    cpu_set_t eligible, selected; CPU_ZERO(&eligible);
    if (sched_getaffinity(0, sizeof(eligible), &eligible)) return 0;
    CPU_ZERO(&selected);
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &eligible)) { CPU_SET(i, &selected); return sched_setaffinity(0, sizeof(selected), &selected) == 0; }
#else
    (void)single_core;
#endif
    return 0;
}
#endif
