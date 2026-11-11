/* Process boundary for MPI: argument vector (no shell), drained stdout,
 * cancellation and timeout kill the owned process group/job. Unit 2. */
#include "hft/api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
int hft_run_mpi_process(const HftConfig *c, HftResult *r, HftAtomic *cancel,
                        HftProgress progress, void *user, const char *worker,
                        int ranks) {
  char cfg[256], prefix[256], result[280], count[32], error[HFT_ERROR] = {0};
  unsigned long token;
#ifdef _WIN32
  token = GetCurrentProcessId();
#else
  token = (unsigned long)getpid();
#endif
  snprintf(cfg, sizeof(cfg), "hft-mpi-%lu-config.json", token);
  snprintf(prefix, sizeof(prefix), "hft-mpi-%lu-result", token);
  snprintf(result, sizeof(result), "%s.json", prefix);
  snprintf(count, sizeof(count), "%d", ranks);
  memset(r, 0, sizeof(*r));
  if (!hft_config_save(c, cfg, r->error))
    return 0;
  double start = hft_now();
  int exitcode = 1;
  char buffer[4096], line[8192];
  size_t used = 0;
#ifdef _WIN32
  /* Reject quoting metacharacters rather than interpreting a user-provided
   * path. */
  if (strchr(worker, '"')) {
    strcpy(r->error, "Invalid worker path");
    remove(cfg);
    return 0;
  }
  SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
  HANDLE rd = NULL, wr = NULL;
  CreatePipe(&rd, &wr, &sa, 0);
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
  STARTUPINFOA si = {0};
  PROCESS_INFORMATION pi = {0};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = wr;
  si.hStdError = wr;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  char command[4096];
  snprintf(command, sizeof(command),
           "mpiexec -n %d \"%s\" --config \"%s\" --out \"%s\"", ranks, worker,
           cfg, prefix);
  HANDLE job = CreateJobObjectA(NULL, NULL);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
                          sizeof(limits));
  if (!CreateProcessA(NULL, command, NULL, NULL, TRUE,
                      CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL, NULL, &si,
                      &pi)) {
    snprintf(r->error, HFT_ERROR, "Cannot start mpiexec (Windows error %lu)",
             GetLastError());
    CloseHandle(rd);
    CloseHandle(wr);
    CloseHandle(job);
    remove(cfg);
    return 0;
  }
  AssignProcessToJobObject(job, pi.hProcess);
  ResumeThread(pi.hThread);
  CloseHandle(wr);
#else
  int pipes[2];
  if (pipe(pipes)) {
    strcpy(r->error, "Cannot create process pipe");
    remove(cfg);
    return 0;
  }
  pid_t pid = fork();
  if (pid < 0) {
    close(pipes[0]);
    close(pipes[1]);
    remove(cfg);
    return 0;
  }
  if (!pid) {
    setpgid(0, 0);
    dup2(pipes[1], STDOUT_FILENO);
    dup2(pipes[1], STDERR_FILENO);
    close(pipes[0]);
    close(pipes[1]);
    execlp("mpiexec", "mpiexec", "-n", count, worker, "--config", cfg, "--out",
           prefix, (char *)NULL);
    _exit(127);
  }
  setpgid(pid, pid);
  close(pipes[1]);
  fcntl(pipes[0], F_SETFL, O_NONBLOCK);
#endif
  for (;;) {
    int n = 0, finished = 0;
#ifdef _WIN32
    DWORD available = 0, readbytes = 0;
    PeekNamedPipe(rd, NULL, 0, NULL, &available, NULL);
    if (available) {
      ReadFile(rd, buffer, sizeof(buffer) - 1, &readbytes, NULL);
      n = (int)readbytes;
    }
    finished = WaitForSingleObject(pi.hProcess, 0) == WAIT_OBJECT_0;
#else
    n = (int)read(pipes[0], buffer, sizeof(buffer) - 1);
    int status;
    pid_t wait = waitpid(pid, &status, WNOHANG);
    if (wait == pid) {
      finished = 1;
      exitcode = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    }
#endif
    if (n > 0) {
      for (int i = 0; i < n; ++i) {
        if (buffer[i] == '\n') {
          line[used] = 0;
          if (progress) {
            HftSnapshot snap = {0};
            snprintf(snap.status, sizeof(snap.status), "%.255s", line);
            const char *fraction = strstr(line, "\"fraction\":");
            if (fraction)
              snap.progress = strtod(fraction + 11, NULL);
            progress(&snap, user);
          }
          used = 0;
        } else if (used < sizeof(line) - 1)
          line[used++] = buffer[i];
      }
    }
    if (finished)
      break;
    if ((cancel && hft_load(cancel)) || hft_now() - start > 300) {
#ifdef _WIN32
      TerminateJobObject(job, 1);
      WaitForSingleObject(pi.hProcess, 1000);
#else
      kill(-pid, SIGTERM);
      waitpid(pid, NULL, 0);
#endif
      r->cancelled = 1;
      strcpy(error, "MPI cancelled or exceeded 300-second timeout");
      break;
    }
    hft_sleep(10);
  }
#ifdef _WIN32
  DWORD code;
  GetExitCodeProcess(pi.hProcess, &code);
  exitcode = (int)code;
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(rd);
  CloseHandle(job);
#else
  close(pipes[0]);
#endif
  remove(cfg);
  int ok = !exitcode && hft_result_import(r, result, error);
  if (!ok)
    snprintf(
        r->error, HFT_ERROR, "%s",
        error[0]
            ? error
            : "MPI worker failed; inspect worker output / MPI installation");
  remove(result);
  snprintf(result, sizeof(result), "%s.csv", prefix);
  remove(result);
  return ok;
}
