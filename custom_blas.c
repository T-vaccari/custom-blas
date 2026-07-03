#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <string.h>

#define DIM 2048

double get_wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void cblas_sgemm_naif(
    enum CBLAS_ORDER     Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int             M,
    int             N,
    int             K,
    float           alpha,
    const float    *A,
    int             lda,
    const float    *B,
    int             ldb,
    float           beta,
    float          *C,
    int             ldc
					){
    // 1. Validate Memory Layout
    if (Order != CblasRowMajor) {
        fprintf(stderr, "cblas_sgemm_naif ERROR: Only CblasRowMajor is supported.\n");
        exit(EXIT_FAILURE); 
    }

    // 2. Validate Transposition
    if (TransA != CblasNoTrans || TransB != CblasNoTrans) {
        fprintf(stderr, "cblas_sgemm_naif ERROR: Matrix transposition is not supported.\n");
        exit(EXIT_FAILURE);
    }

    // 3. Validate Dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "cblas_sgemm_naif ERROR: Matrix dimensions (M, N, K) must be positive.\n");
        exit(EXIT_FAILURE);
    }
	
	for(size_t i = 0 ; i < N ; i++){
		for(size_t j = 0 ; j < N; j++){
			
			float acc = 0;
			for (size_t k = 0; k < N; k++){
				acc += *(A + lda*i + k) * *(B + j + ldb * k);
			}
			
			*(C + i * ldc + j) = acc;
		}
	}
}

void cblas_sgemm_naif_reorganized(
    enum CBLAS_ORDER     Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int             M,
    int             N,
    int             K,
    float           alpha,
    const float    *A,
    int             lda,
    const float    *B,
    int             ldb,
    float           beta,
    float          *C,
    int             ldc
					){
    // 1. Validate Memory Layout
    if (Order != CblasRowMajor) {
        fprintf(stderr, "cblas_sgemm_naif ERROR: Only CblasRowMajor is supported.\n");
        exit(EXIT_FAILURE); 
    }

    // 2. Validate Transposition
    if (TransA != CblasNoTrans || TransB != CblasNoTrans) {
        fprintf(stderr, "cblas_sgemm_naif ERROR: Matrix transposition is not supported.\n");
        exit(EXIT_FAILURE);
    }

    // 3. Validate Dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "cblas_sgemm_naif ERROR: Matrix dimensions (M, N, K) must be positive.\n");
        exit(EXIT_FAILURE);
    }
	
	for(size_t i = 0; i < N; i++){
		for(size_t k = 0; k < N ; k++){
			float a_i_k = *(A + lda * i + k);
			for(size_t j = 0; j < N ; j++){
				*(C + i*ldc + j) += a_i_k * *(B+ldb*k+j);
			}
		}
	}	
}

void cblas_sgemm_vectorized_pragma(
    enum CBLAS_ORDER     Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int             M,
    int             N,
    int             K,
    float           alpha,
    const float    *A,
    int             lda,
    const float    *B,
    int             ldb,
    float           beta,
    float          *C,
    int             ldc
					){
    // 1. Validate Memory Layout
    if (Order != CblasRowMajor) {
        fprintf(stderr, "cblas_sgemm_vectorized_pragma ERROR: Only CblasRowMajor is supported.\n");
        exit(EXIT_FAILURE); 
    }

    // 2. Validate Transposition
    if (TransA != CblasNoTrans || TransB != CblasNoTrans) {
        fprintf(stderr, "cblas_sgemm_vectorized_pragma ERROR: Matrix transposition is not supported.\n");
        exit(EXIT_FAILURE);
    }

    // 3. Validate Dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "cblas_sgemm_vectorized_pragma ERROR: Matrix dimensions (M, N, K) must be positive.\n");
        exit(EXIT_FAILURE);
    }
	
	for(size_t i = 0; i < N; i++){
		for(size_t k = 0; k < N ; k++){
			float a_i_k = *(A + lda * i + k);
			
			#pragma clang loop vectorize(enable) interleave(enable)
			for(size_t j = 0; j < N ; j++){
				*(C + i*ldc + j) += a_i_k * *(B+ldb*k+j);
			}
		}
	}	
}

void fill_random_matrix(float *mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(mat + i * ld + j) = (float)rand() / (float)RAND_MAX;
        }
    }
}

void print_matrix(const char *label, const float *mat, int rows, int cols, int ld) {
    printf("--- %s (%dx%d) ---\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.3f ", *(mat + i * ld + j));
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    int M = DIM;
    int N = DIM;
    int K = DIM;

    int lda = K;
    int ldb = N;
    int ldc = N;

    float alpha = 1.0f;
    float beta  = 0.0f;

    float *A = (float *)malloc(M * lda * sizeof(float));
    float *B = (float *)malloc(K * ldb * sizeof(float));
    float *C = (float *)malloc(M * ldc * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Fatal: Memory allocation failed.\n");
        free(A); free(B); free(C);
        return EXIT_FAILURE;
    }

    fill_random_matrix(A, M, K, lda);
    fill_random_matrix(B, K, N, ldb);
	
    // 6. Execute Naive GEMM
	double total_ops = 2.0 * (double)M * (double)N * (double)K;
	double start = get_wall_time();
	cblas_sgemm_naif(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
	double end = get_wall_time();
	double elapsed_seconds = end - start;
	double gflops = (total_ops / elapsed_seconds) / 1e9;
	printf("---NAIF Version---\n");
	printf("Time: %f seconds\n", elapsed_seconds);
	printf("Performance: %f GFLOPS\n", gflops);

	memset(C, 0, M * ldc * sizeof(float));
	// 7. NAIF reorganized Version	
	start = get_wall_time();
	cblas_sgemm_naif_reorganized(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
	end = get_wall_time();
	elapsed_seconds = end - start;
	gflops = (total_ops / elapsed_seconds) / 1e9;
	printf("---NAIF reorganized Version---\n");
	printf("Time: %f seconds\n", elapsed_seconds);
	printf("Performance: %f GFLOPS\n", gflops);

	memset(C, 0, M * ldc * sizeof(float));
	// 8. Vectorized Pragma Version	
	start = get_wall_time();
	cblas_sgemm_vectorized_pragma(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
	end = get_wall_time();
	elapsed_seconds = end - start;
	gflops = (total_ops / elapsed_seconds) / 1e9;
	printf("---Vectorized Pragma Version---\n");
	printf("Time: %f seconds\n", elapsed_seconds);
	printf("Performance: %f GFLOPS\n", gflops);

	printf("Sanity Check (prevents compiler DCE): %f\n", C[0]);
    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
