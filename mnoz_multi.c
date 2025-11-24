#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <inttypes.h>

#define N 1000

/* Uwaga:
   Program alokuje duże tablice statycznie (około kilkadziesiąt MB).
   Dostosuj N jeśli brakuje pamięci. */

static int    A_i[N][N], B_i[N][N];
static long   C_i[N][N];

static double A_d[N][N], B_d[N][N], C_d[N][N];

double timediff(struct timespec a, struct timespec b){
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec)/1e9;
}

void fill_int_matrix(int A[N][N]) {
    int value = 1;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            A[i][j] = value++ % 100 + 1;
}

void fill_double_matrix(double A[N][N]) {
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            A[i][j] = sin((i*N + j) * M_PI / 180.0);
}

void matmul_int_single(void) {
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) {
            long sum = 0;
            for(int k=0;k<N;k++)
                sum += (long)A_i[i][k] * (long)B_i[k][j];
            C_i[i][j] = sum;
        }
}

void matmul_double_single(void) {
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) {
            double sum = 0.0;
            for(int k=0;k<N;k++)
                sum += A_d[i][k] * B_d[k][j];
            C_d[i][j] = sum;
        }
}

/* Helper: write_n bytes (handle partial writes) */
ssize_t write_n(int fd, const void *buf, size_t n) {
    const char *p = buf;
    size_t left = n;
    while(left > 0) {
        ssize_t w = write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        left -= w;
        p += w;
    }
    return (ssize_t)n;
}

/* Helper: read_n bytes (handle partial reads) */
ssize_t read_n(int fd, void *buf, size_t n) {
    char *p = buf;
    size_t left = n;
    while(left > 0) {
        ssize_t r = read(fd, p, left);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) break; // EOF
        left -= r;
        p += r;
    }
    return (ssize_t)(n - left);
}

/* Parallel version for int matrices using fork() and pipes.
   parent creates `workers` children; for each child a pipe[2] is created
   (child writes computed rows to parent). Each child computes rows from
   start_row (inclusive) to end_row (exclusive). */
int matmul_int_forked(int workers) {
    if (workers < 1) workers = 1;
    pid_t *pids = malloc(sizeof(pid_t) * workers);
    int (*pipes)[2] = malloc(sizeof(int[2]) * workers);
    if (!pids || !pipes) { perror("malloc"); return -1; }

    int rows_per = N / workers;
    int extra = N % workers;

    for(int w=0; w<workers; ++w) {
        if (pipe(pipes[w]) < 0) {
            perror("pipe");
            return -1;
        }
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return -1;
        }
        if (pid == 0) {
            /* child */
            close(pipes[w][0]); // close read end
            /* close other pipes' fds inherited */
            for(int t=0;t<w;++t){
                close(pipes[t][0]);
                close(pipes[t][1]);
            }
            int start = w * rows_per + (w < extra ? w : extra);
            int count = rows_per + (w < extra ? 1 : 0);
            int end = start + count;
            /* compute assigned rows */
            for(int i=start; i<end; ++i) {
                for(int j=0;j<N;++j) {
                    long sum = 0;
                    for(int k=0;k<N;++k)
                        sum += (long)A_i[i][k] * (long)B_i[k][j];
                    C_i[i][j] = sum; // write into child's memory
                }
                /* send this row to parent via pipe */
                ssize_t bytes = sizeof(long) * (size_t)N;
                if (write_n(pipes[w][1], &C_i[i][0], (size_t)bytes) != bytes) {
                    perror("write_n child");
                    /* even if error, exit child */
                    close(pipes[w][1]);
                    _exit(1);
                }
            }
            close(pipes[w][1]);
            _exit(0);
        } else {
            /* parent */
            pids[w] = pid;
            close(pipes[w][1]); // parent closes write end, keeps read end
        }
    }

    /* parent: read data from pipes into C_i */
    for(int w=0; w<workers; ++w) {
        int start = w * rows_per + (w < extra ? w : extra);
        int count = rows_per + (w < extra ? 1 : 0);
        int end = start + count;
        for(int i=start; i<end; ++i) {
            ssize_t bytes = sizeof(long) * (size_t)N;
            ssize_t rr = read_n(pipes[w][0], &C_i[i][0], (size_t)bytes);
            if (rr != bytes) {
                fprintf(stderr, "Parent: read expected %zd got %zd for row %d (pipe %d)\n", bytes, rr, i, w);
                /* continue reading other rows */
            }
        }
        close(pipes[w][0]);
    }

    /* wait for all children */
    int status = 0;
    for(int w=0; w<workers; ++w) {
        pid_t wp = waitpid(pids[w], &status, 0);
        if (wp < 0) perror("waitpid");
        else {
            if (WIFEXITED(status)) {
                if (WEXITSTATUS(status) != 0) {
                    fprintf(stderr, "Child %d exited with code %d\n", (int)pids[w], WEXITSTATUS(status));
                }
            } else if (WIFSIGNALED(status)) {
                fprintf(stderr, "Child %d killed by signal %d\n", (int)pids[w], WTERMSIG(status));
            }
        }
    }

    free(pids);
    free(pipes);
    return 0;
}

/* Parallel version for double matrices using same approach */
int matmul_double_forked(int workers) {
    if (workers < 1) workers = 1;
    pid_t *pids = malloc(sizeof(pid_t) * workers);
    int (*pipes)[2] = malloc(sizeof(int[2]) * workers);
    if (!pids || !pipes) { perror("malloc"); return -1; }

    int rows_per = N / workers;
    int extra = N % workers;

    for(int w=0; w<workers; ++w) {
        if (pipe(pipes[w]) < 0) {
            perror("pipe");
            return -1;
        }
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return -1;
        }
        if (pid == 0) {
            /* child */
            close(pipes[w][0]); // close read end
            /* close other pipes' fds inherited */
            for(int t=0;t<w;++t){
                close(pipes[t][0]);
                close(pipes[t][1]);
            }
            int start = w * rows_per + (w < extra ? w : extra);
            int count = rows_per + (w < extra ? 1 : 0);
            int end = start + count;
            /* compute assigned rows */
            for(int i=start; i<end; ++i) {
                for(int j=0;j<N;++j) {
                    double sum = 0.0;
                    for(int k=0;k<N;++k)
                        sum += A_d[i][k] * B_d[k][j];
                    C_d[i][j] = sum;
                }
                /* send this row to parent via pipe */
                ssize_t bytes = sizeof(double) * (size_t)N;
                if (write_n(pipes[w][1], &C_d[i][0], (size_t)bytes) != bytes) {
                    perror("write_n child double");
                    close(pipes[w][1]);
                    _exit(1);
                }
            }
            close(pipes[w][1]);
            _exit(0);
        } else {
            /* parent */
            pids[w] = pid;
            close(pipes[w][1]); // parent closes write end
        }
    }

    /* parent: read data from pipes into C_d */
    for(int w=0; w<workers; ++w) {
        int start = w * rows_per + (w < extra ? w : extra);
        int count = rows_per + (w < extra ? 1 : 0);
        int end = start + count;
        for(int i=start; i<end; ++i) {
            ssize_t bytes = sizeof(double) * (size_t)N;
            ssize_t rr = read_n(pipes[w][0], &C_d[i][0], (size_t)bytes);
            if (rr != bytes) {
                fprintf(stderr, "Parent double: read expected %zd got %zd for row %d (pipe %d)\n", bytes, rr, i, w);
            }
        }
        close(pipes[w][0]);
    }

    /* wait for all children */
    int status = 0;
    for(int w=0; w<workers; ++w) {
        pid_t wp = waitpid(pids[w], &status, 0);
        if (wp < 0) perror("waitpid");
        else {
            if (WIFEXITED(status)) {
                if (WEXITSTATUS(status) != 0) {
                    fprintf(stderr, "Child %d exited with code %d\n", (int)pids[w], WEXITSTATUS(status));
                }
            } else if (WIFSIGNALED(status)) {
                fprintf(stderr, "Child %d killed by signal %d\n", (int)pids[w], WTERMSIG(status));
            }
        }
    }

    free(pids);
    free(pipes);
    return 0;
}

/* quick checksum helpers to verify results (sum of all elements) */
long checksum_int(void) {
    long s = 0;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            s += C_i[i][j];
    return s;
}

double checksum_double(void) {
    double s = 0.0;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            s += C_d[i][j];
    return s;
}

int main(void) {
    printf("Mnozenie macierzy %dx%d - test jednowatkowy vs fork()+pipe\n", N, N);
    /* fill matrices */
    fill_int_matrix(A_i);
    fill_int_matrix(B_i);
    fill_double_matrix(A_d);
    fill_double_matrix(B_d);

    struct timespec t1, t2;

    /* SINGLE-PROCESS (int) */
    memset(C_i, 0, sizeof(C_i));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    matmul_int_single();
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double t_single_int = timediff(t1,t2);
    long cs_int_single = checksum_int();
    printf("Jednowatkowo (int): %.6f s, checksum=%" PRId64 "\n", t_single_int, (int64_t)cs_int_single);

    /* FORK+PIPE (int) */
    memset(C_i, 0, sizeof(C_i));
    int workers = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (workers < 1) workers = 1;
    printf("Uzyje %d workerow (fork) do liczenia int.\n", workers);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (matmul_int_forked(workers) != 0) {
        fprintf(stderr, "matmul_int_forked failed\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double t_fork_int = timediff(t1,t2);
    long cs_int_fork = checksum_int();
    printf("Fork+pipe (int): %.6f s, checksum=%" PRId64 "\n", t_fork_int, (int64_t)cs_int_fork);

    /* VERIFY checksums match */
    if (cs_int_single != cs_int_fork) {
        fprintf(stderr, "Uwaga: checksumy int sie roznia! single=%" PRId64 " fork=%" PRId64 "\n",
                (int64_t)cs_int_single, (int64_t)cs_int_fork);
    }

    /* SINGLE-PROCESS (double) */
    memset(C_d, 0, sizeof(C_d));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    matmul_double_single();
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double t_single_double = timediff(t1,t2);
    double cs_double_single = checksum_double();
    printf("Jednowatkowo (double): %.6f s, checksum=%e\n", t_single_double, cs_double_single);

    /* FORK+PIPE (double) */
    memset(C_d, 0, sizeof(C_d));
    printf("Uzyje %d workerow (fork) do liczenia double.\n", workers);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (matmul_double_forked(workers) != 0) {
        fprintf(stderr, "matmul_double_forked failed\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double t_fork_double = timediff(t1,t2);
    double cs_double_fork = checksum_double();
    printf("Fork+pipe (double): %.6f s, checksum=%e\n", t_fork_double, cs_double_fork);

    if (fabs(cs_double_single - cs_double_fork) > 1e-6) {
        fprintf(stderr, "Uwaga: checksumy double sie roznia!\n");
    }

    /* Summary */
    printf("\n--- Podsumowanie ---\n");
    printf("INT: single=%.6f s, fork=%.6f s, speedup=%.3fx\n", t_single_int, t_fork_int, t_single_int / t_fork_int);
    printf("DBL: single=%.6f s, fork=%.6f s, speedup=%.3fx\n", t_single_double, t_fork_double, t_single_double / t_fork_double);

    return 0;
}

