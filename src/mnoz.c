#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100

void fill_int_matrix(int A[N][N]) {
    int value = 1;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            A[i][j] = value++ % 100 + 1;
}

void fill_double_matrix(double A[N][N]) {
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            A[i][j] = sin((i*N + j) * M_PI / 180.0);   // stopnie -> radiany
}

void matmul_int(int A[N][N], int B[N][N], long C[N][N]) {
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) {
            long sum = 0;
            for(int k=0;k<N;k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

void matmul_double(double A[N][N], double B[N][N], double C[N][N]) {
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) {
            double sum = 0.0;
            for(int k=0;k<N;k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

double timediff(struct timespec a, struct timespec b){
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec)/1e9;
}

int main(){
	printf("Program wykonuje mnożenie dwóch macierzy\n"); // Zmiana
    static int    A_i[N][N], B_i[N][N];
    static long   C_i[N][N];

    static double A_d[N][N], B_d[N][N], C_d[N][N];

    fill_int_matrix(A_i);
    fill_int_matrix(B_i);

    fill_double_matrix(A_d);
    fill_double_matrix(B_d);

    struct timespec t1, t2;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    matmul_int(A_i, B_i, C_i);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("Czas (int, bez NEON): %.6f s\n", timediff(t1,t2));

    clock_gettime(CLOCK_MONOTONIC, &t1);
    matmul_double(A_d, B_d, C_d);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("Czas (double, bez NEON): %.6f s\n", timediff(t1,t2));

    return 0;
}

