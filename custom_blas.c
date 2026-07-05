#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <string.h>
#include <assert.h>

//Utils to get the current time
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

    // 4. This implementation only iterates using N in all three loop bounds
    //    (see below), so it is only correct for square matrices.

    assert(M == N && N == K && "cblas_sgemm_naif only supports square matrices (M == N == K)");
	
	for(size_t i = 0 ; i < N ; i++){
		for(size_t j = 0 ; j < N; j++){
			float acc = 0;
			for (size_t k = 0; k < N; k++){
				//Notice here how we could also squeeze more by pre-calculating the costant lda*i for the
				//entire for
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
        fprintf(stderr, "cblas_sgemm_naif_reorganized ERROR: Only CblasRowMajor is supported.\n");
        exit(EXIT_FAILURE); 
    }

    // 2. Validate Transposition
    if (TransA != CblasNoTrans || TransB != CblasNoTrans) {
        fprintf(stderr, "cblas_sgemm_naif_reorganized ERROR: Matrix transposition is not supported.\n");
        exit(EXIT_FAILURE);
    }

    // 3. Validate Dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "cblas_sgemm_naif_reorganized ERROR: Matrix dimensions (M, N, K) must be positive.\n");
        exit(EXIT_FAILURE);
    }

    // 4. Same N-only loop bounds as cblas_sgemm_naif above: square-only.
    assert(M == N && N == K && "cblas_sgemm_naif_reorganized only supports square matrices (M == N == K)");
	
	for(size_t i = 0; i < N; i++){
		for(size_t k = 0; k < N ; k++){
			float a_i_k = *(A + lda * i + k);

			// Explicitly forbid the compiler from autovectorizing this loop,
			// even though its access pattern (stride-1 on B and C) is
			// otherwise vectorizable. This isolates the effect of loop
			// reordering (i-k-j vs i-j-k) from the effect of SIMD, so it
			// can be compared separately against the vectorized version
			// below. unroll(disable) is added too, since aggressive
			// unrolling at -O3 can partially recover SIMD-like throughput
			// even without formal vectorization.
			#pragma clang loop vectorize(disable) unroll(disable)
			for(size_t j = 0; j < N ; j++){
				*(C + i*ldc + j) += a_i_k * *(B+ldb*k+j);
			}
		}
	}
	//In this version the key insight is that we can compute the same thing but witouht jumping between rows of B, when we 
	//have loaded one row we maximize flop per byte moved and then we can move to the next row.
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

    // 4. Same N-only loop bounds as the other naive versions: square-only.
    assert(M == N && N == K && "cblas_sgemm_vectorized_pragma only supports square matrices (M == N == K)");
	
	for(size_t i = 0; i < N; i++){
		for(size_t k = 0; k < N ; k++){
			float a_i_k = *(A + lda * i + k);
			//It's the same version as before but we the adjustment that we are asking to the compiler to vectorize
			//that instruction, thanks to the 4 words(128 bit) NEON register of the macbook m2
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

// Generic timing/reporting harness so every version (naive or real BLAS)
// is measured identically. fn_name is stored purely for the printout.
typedef void (*gemm_fn_t)(
    enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
    int, int, int, float, const float *, int, const float *, int,
    float, float *, int
);

void run_benchmark(
    const char *label,
    gemm_fn_t fn,
    int M, int N, int K,
    float alpha, const float *A, int lda,
    const float *B, int ldb,
    float beta, float *C, int ldc,
    double total_ops
){
    memset(C, 0, (size_t)M * (size_t)ldc * sizeof(float));

    double start = get_wall_time();
    fn(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    double end = get_wall_time();

    double elapsed_seconds = end - start;
    double gflops = (total_ops / elapsed_seconds) / 1e9;

    printf("--- %s ---\n", label);
    printf("Time: %f seconds\n", elapsed_seconds);
    printf("Performance: %f GFLOPS\n", gflops);
    printf("Sanity Check (prevents compiler DCE): %f\n\n", C[0]);
}

// Runs all four GEMM variants for a single square dimension N.
void run_all_variants_for_size(int N) {
    int M = N;
    int K = N;

    int lda = K;
    int ldb = N;
    int ldc = N;

    float alpha = 1.0f;
    float beta  = 0.0f;

    float *A = (float *)malloc((size_t)M * (size_t)lda * sizeof(float));
    float *B = (float *)malloc((size_t)K * (size_t)ldb * sizeof(float));
    float *C = (float *)malloc((size_t)M * (size_t)ldc * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Fatal: Memory allocation failed for N=%d.\n", N);
        free(A); free(B); free(C);
        exit(EXIT_FAILURE);
    }

    fill_random_matrix(A, M, K, lda);
    fill_random_matrix(B, K, N, ldb);

    double total_ops = 2.0 * (double)M * (double)N * (double)K;

    printf("\n=========================================\n");
    printf("  N = %d\n", N);
    printf("=========================================\n");

    // 1. Naive version (i-j-k order, no accumulator reuse across cache lines)
    run_benchmark("NAIVE Version", cblas_sgemm_naif,
                  M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, total_ops);

    // 2. Reorganized naive (i-k-j order, sequential access on B and C rows),
    //    but with autovectorization explicitly disabled. This isolates the
    //    effect of loop reordering / cache locality alone, since otherwise
    //    the compiler would autovectorize this too (same stride-1 access
    //    pattern as the pragma version below),
    run_benchmark("NAIVE Reorganized Version (scalar, no autovec)", cblas_sgemm_naif_reorganized,
                  M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, total_ops);

    // 3. Same loop order as above, now with vectorization explicitly
    //    enabled via pragma. Compare directly against stage 2 to see the
    //    SIMD contribution in isolation from the loop-reordering
    //    contribution already captured above.
    run_benchmark("Vectorized Pragma Version (reordered + SIMD)", cblas_sgemm_vectorized_pragma,
                  M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, total_ops);

    // 4. Real Accelerate cblas_sgemm: cache-blocked, SIMD/AMX-accelerated
    //    reference implementation.     memset(C, 0, (size_t)M * (size_t)ldc * sizeof(float));
    double start = get_wall_time();
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
    double end = get_wall_time();
    double elapsed_seconds = end - start;
    double gflops = (total_ops / elapsed_seconds) / 1e9;
    printf("--- Accelerate cblas_sgemm (REAL BLAS) ---\n");
    printf("Time: %f seconds\n", elapsed_seconds);
    printf("Performance: %f GFLOPS\n", gflops);
    printf("Sanity Check (prevents compiler DCE): %f\n", C[0]);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    // Sweep across these square dimensions. Edit freely; each entry runs
    // all four GEMM variants independently with its own fresh buffers.
    int sizes[] = { 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        run_all_variants_for_size(sizes[s]);
    }

    return EXIT_SUCCESS;
}
