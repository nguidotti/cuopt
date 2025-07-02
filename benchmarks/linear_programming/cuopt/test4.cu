/**********************************************************************
 *  Three cuSPARSE SpMM variants that all deliver the same column-major
 *  result C (4 × 2) but use different dense-matrix layouts internally.
 *
 *    1)  B = COL,  C = COL  (reference code)
 *    2)  B = ROW,  C = ROW  (transpose C back to COL on the host)
 *    3)  B = ROW,  C = COL  (transpose B on the host before SpMM)
 *    4)  B = COL,  C = ROW  (transpose C back to COL on the host)
 *
 *  All three functions take exactly the same column-major B as input
 *  and return C in column-major layout.  The body of each function is
 *  self-contained; all required transposes happen inside the function.
 *********************************************************************/

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include "benchmark_helper.hpp"
#include <raft/core/handle.hpp>
#include <raft/sparse/linalg/transpose.cuh>

/* ------------------------------------------------------------------ */
/*  error checking helpers                                             */
#define CHECK_CUDA(call)                                                     \
{                                                                            \
    cudaError_t _status = (call);                                            \
    if (_status != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(_status));            \
        return EXIT_FAILURE;                                                 \
    }                                                                        \
}

#define CHECK_CUSPARSE(call)                                                 \
{                                                                            \
    cusparseStatus_t _status = (call);                                       \
    if (_status != CUSPARSE_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuSPARSE error %s:%d  %s\n",                        \
                __FILE__, __LINE__, cusparseGetErrorString(_status));        \
        return EXIT_FAILURE;                                                 \
    }                                                                        \
}

/* ================================================================== */
/*  helper: transpose CSR matrix using RAFT on device                 */
static void transpose_csr_matrix_device(const raft::handle_t* handle,
                                        int A_rows, int A_cols, int A_nnz,
                                        const int *dA_csrOffsets, const int *dA_columns, const double *dA_values,
                                        int *dAT_csrOffsets, int *dAT_columns, double *dAT_values)
{
    raft::sparse::linalg::csr_transpose(*handle,
                                       const_cast<int*>(dA_csrOffsets),
                                       const_cast<int*>(dA_columns),
                                       const_cast<double*>(dA_values),
                                       dAT_csrOffsets,
                                       dAT_columns,
                                       dAT_values,
                                       A_rows,
                                       A_cols,
                                       A_nnz,
                                       handle->get_stream());
}



/* ================================================================== */
/*  helper: create, run SpMM, copy result                             */
static float run_spmm(bool B_row_major,
                    bool C_row_major,
                    bool transpose_A,
                    const double *hB_in,   /* column-major input       */
                    double       *hC_out,  /* column-major output      */
                    int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                    const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                    int B_NUM_ROWS, int B_NUM_COLS,
                    const raft::handle_t* raft_handle)
{
    std::string scope_name = "run_spmm with ";
    scope_name += B_row_major ? "B row-major" : "B col-major";
    scope_name += " and ";
    scope_name += C_row_major ? "C row-major" : "C col-major";
    scope_name += " and ";
    scope_name += transpose_A ? "transpose_A" : "no transpose_A";

    const int num_iterations = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA( cudaEventCreate(&start) )
    CHECK_CUDA( cudaEventCreate(&stop) );
    float total_time_ms = 0.0;

    double alpha = 1.f, beta = 0.f;
    rmm::device_scalar<double> alpha_scalar(alpha, raft_handle->get_stream());
    rmm::device_scalar<double> beta_scalar(beta, raft_handle->get_stream());

    for (int i = 0; i < num_iterations; i++) {
    raft::common::nvtx::range fun_scope{scope_name.c_str()};

    float local_time_ms = 0.0;

    /* ---------- device allocations ---------------------------------- */
    int   B_size = B_NUM_ROWS * B_NUM_COLS;
    int   C_size_final = (transpose_A ? A_NUM_COLS : A_NUM_ROWS) * B_NUM_COLS;

    rmm::device_uvector<int> dA_csrOffsets_vec(A_NUM_ROWS+1, raft_handle->get_stream());
    rmm::device_uvector<int> dA_columns_vec(A_NNZ, raft_handle->get_stream());
    rmm::device_uvector<double> dA_values_vec(A_NNZ, raft_handle->get_stream());
    rmm::device_uvector<double> dB_vec(B_size, raft_handle->get_stream());
    rmm::device_uvector<double> dC_vec(C_size_final, raft_handle->get_stream());
    rmm::device_uvector<double> dB_transposed_vec(B_size, raft_handle->get_stream());
    rmm::device_uvector<double> dC_transposed_vec(C_size_final, raft_handle->get_stream());

    int   *dA_csrOffsets = dA_csrOffsets_vec.data();
    int   *dA_columns = dA_columns_vec.data();
    double *dA_values = dA_values_vec.data();
    double *dB = dB_vec.data();
    double *dC = dC_vec.data();

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_NUM_ROWS+1)*sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                           A_NNZ*sizeof(int),          cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dA_values,  hA_values,
                           A_NNZ*sizeof(double),        cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB,         hB_in,
                           B_size*sizeof(double),       cudaMemcpyHostToDevice) );

    /* ---------- Step 0.5: if required, transpose A on device -------- */
    int *dA_final_csrOffsets = dA_csrOffsets;
    int *dA_final_columns = dA_columns;
    double *dA_final_values = dA_values;
    int A_final_rows = A_NUM_ROWS;
    int A_final_cols = A_NUM_COLS;

    rmm::device_uvector<int> dAT_csrOffsets_vec(0, raft_handle->get_stream());
    rmm::device_uvector<int> dAT_columns_vec(0, raft_handle->get_stream());
    rmm::device_uvector<double> dAT_values_vec(0, raft_handle->get_stream());

    if (transpose_A) {
        /* Create device vectors for A^T */
        dAT_csrOffsets_vec.resize(A_NUM_COLS+1, raft_handle->get_stream());
        dAT_columns_vec.resize(A_NNZ, raft_handle->get_stream());
        dAT_values_vec.resize(A_NNZ, raft_handle->get_stream());

        /* Transpose A on device using RAFT */
        transpose_csr_matrix_device(raft_handle, A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                                   dA_csrOffsets, dA_columns, dA_values,
                                   dAT_csrOffsets_vec.data(), dAT_columns_vec.data(), dAT_values_vec.data());

        /* Use A^T for SpMM */
        dA_final_csrOffsets = dAT_csrOffsets_vec.data();
        dA_final_columns = dAT_columns_vec.data();
        dA_final_values = dAT_values_vec.data();
        A_final_rows = A_NUM_COLS;  /* A^T dimensions */
        A_final_cols = A_NUM_ROWS;
    }

    /* ---------- Step 0: if required, transpose B on the device -------- */
    int   ldb   = 0;
    cusparseOrder_t orderB;

    if (B_row_major) {
        raft::common::nvtx::range fun_scope{"transpose B"};

        float b_transpose_time_ms = 0.0;
        CHECK_CUDA( cudaEventRecord(start, raft_handle->get_stream()) );
        /* transpose B on device using cuBLAS */
        double *dB_transposed = dB_transposed_vec.data();
        RAFT_CUBLAS_TRY( cublasDgeam(raft_handle->get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                  B_NUM_COLS, B_NUM_ROWS,
                                  alpha_scalar.data(), dB, B_NUM_ROWS,
                                  beta_scalar.data(), dB_transposed, B_NUM_COLS,
                                  dB_transposed, B_NUM_COLS) );
        CHECK_CUDA( cudaEventRecord(stop, raft_handle->get_stream()) );
        CHECK_CUDA( cudaEventSynchronize(stop) );
        CHECK_CUDA( cudaEventElapsedTime(&b_transpose_time_ms, start, stop) );
        local_time_ms += b_transpose_time_ms;

        dB = dB_transposed;
        ldb     = B_NUM_COLS;                     /* stride between rows */
        orderB  = CUSPARSE_ORDER_ROW;
    } else {
        ldb     = B_NUM_ROWS;                     /* stride between cols */
        orderB  = CUSPARSE_ORDER_COL;
    }

    /* ---------- cuSPARSE descriptors --------------------------------- */
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_final_rows, A_final_cols, A_NNZ,
                                      dA_final_csrOffsets, dA_final_columns, dA_final_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB,
                       B_NUM_ROWS, B_NUM_COLS, ldb,
                       dB, CUDA_R_64F, orderB) );

    int   ldc  = C_row_major ? B_NUM_COLS : A_final_rows;
    cusparseOrder_t orderC = C_row_major ? CUSPARSE_ORDER_ROW
                                         : CUSPARSE_ORDER_COL;

    CHECK_CUSPARSE( cusparseCreateDnMat(&matC,
                       A_final_rows, B_NUM_COLS, ldc,
                       dC, CUDA_R_64F, orderC) );

    /* ---------- SpMM -------------------------------------------------- */
    size_t bufSize = 0;

    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                        raft_handle->get_cusparse_handle(),
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        alpha_scalar.data(), matA, matB, beta_scalar.data(), matC,
                        CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG3, &bufSize) );

    rmm::device_uvector<char> dBuffer_vec(bufSize, raft_handle->get_stream());
    void *dBuffer = dBuffer_vec.data();



    float spmm_time_ms = 0.0;
    CHECK_CUDA( cudaEventRecord(start, raft_handle->get_stream()) );
    {
        raft::common::nvtx::range fun_scope{"SpMM"};
    CHECK_CUSPARSE( cusparseSpMM(raft_handle->get_cusparse_handle(),
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        alpha_scalar.data(), matA, matB, beta_scalar.data(), matC,
                        CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG3, dBuffer) );
    CHECK_CUDA( cudaEventRecord(stop, raft_handle->get_stream()) );
    CHECK_CUDA( cudaEventSynchronize(stop) );
    CHECK_CUDA( cudaEventElapsedTime(&spmm_time_ms, start, stop) );
    local_time_ms += spmm_time_ms;
    }

    /* ---------- copy result back ------------------------------------- */
    if (C_row_major) {
        /* transpose C on device using cuBLAS */
        raft::common::nvtx::range fun_scope{"transpose C"};
        double *dC_transposed = dC_transposed_vec.data();
        int mC = A_final_rows;
        int nC = B_NUM_COLS;

        float c_transpose_time_ms = 0.0;
        CHECK_CUDA( cudaEventRecord(start, raft_handle->get_stream()) );

        RAFT_CUBLAS_TRY( cublasDgeam(raft_handle->get_cublas_handle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    mC,                 // rows  of result (= nC of op(A))
                    nC,                 // cols  of result (= mC of op(A))
                    alpha_scalar.data(),
                    dC,  nC,            // lda = nC for row-major A
                    beta_scalar.data(),
                    nullptr, mC,        // B not used
                    dC_transposed, mC) ); // ldc = mC for column-major C

        CHECK_CUDA( cudaEventRecord(stop, raft_handle->get_stream()) );
        CHECK_CUDA( cudaEventSynchronize(stop) );
        CHECK_CUDA( cudaEventElapsedTime(&c_transpose_time_ms, start, stop) );
        local_time_ms += c_transpose_time_ms;
        CHECK_CUDA( cudaMemcpy(hC_out, dC_transposed, C_size_final*sizeof(double),
                               cudaMemcpyDeviceToHost) );
    } else {
        CHECK_CUDA( cudaMemcpy(hC_out, dC, C_size_final*sizeof(double),
                               cudaMemcpyDeviceToHost) );
    }

    total_time_ms += local_time_ms;
    /* ---------- clean-up --------------------------------------------- */
    /* device_uvector automatically manages memory - no need for cudaFree */
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) );
    }

    total_time_ms /= num_iterations;

    CHECK_CUDA( cudaEventDestroy(start) );
    CHECK_CUDA( cudaEventDestroy(stop) );


    return total_time_ms;
}

/* ================================================================== */
/*  public wrappers demanded by the user                               */
float spmm_col_col(const double *hB_col_in, double *hC_out,
                 int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                 const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                 int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/false,
                    /*C_row_major=*/false,
                    /*transpose_A=*/false,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

float spmm_row_row(const double *hB_col_in, double *hC_out,
                 int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                 const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                 int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/true,
                    /*C_row_major=*/true,
                    /*transpose_A=*/false,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

float spmm_rowcol (const double *hB_col_in, double *hC_out,
                 int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                 const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                 int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/true,
                    /*C_row_major=*/false,
                    /*transpose_A=*/false,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

float spmm_col_row(const double *hB_col_in, double *hC_out,
                 int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                 const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                 int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/false,
                    /*C_row_major=*/true,
                    /*transpose_A=*/false,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

/* ================================================================== */
/*  A^T * B variants - manually transpose A then do SpMM              */
float spmm_AT_col_col(const double *hB_col_in, double *hC_out,
                    int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                    const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                    int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/false,
                    /*C_row_major=*/false,
                    /*transpose_A=*/true,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

float spmm_AT_row_row(const double *hB_col_in, double *hC_out,
                    int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                    const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                    int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/true,
                    /*C_row_major=*/true,
                    /*transpose_A=*/true,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

float spmm_AT_rowcol(const double *hB_col_in, double *hC_out,
                   int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                   const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                   int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/true,
                    /*C_row_major=*/false,
                    /*transpose_A=*/true,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

float spmm_AT_col_row(const double *hB_col_in, double *hC_out,
                    int A_NUM_ROWS, int A_NUM_COLS, int A_NNZ,
                    const int *hA_csrOffsets, const int *hA_columns, const double *hA_values,
                    int B_NUM_ROWS, int B_NUM_COLS, const raft::handle_t* raft_handle)
{
    return run_spmm(/*B_row_major=*/false,
                    /*C_row_major=*/true,
                    /*transpose_A=*/true,
                    hB_col_in, hC_out,
                    A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets, hA_columns, hA_values,
                    B_NUM_ROWS, B_NUM_COLS, raft_handle);
}

/* ================================================================== */
/*  CPU reference SpMM: C = A * B (A sparse CSR, B and C dense col-major) */
static void cpu_spmm_csr(int A_rows, int A_cols, int A_nnz,
                        const int *A_csrOffsets, const int *A_columns, const double *A_values,
                        const double *B, int B_rows, int B_cols,
                        double *C)
{
    // Initialize C to zero
    for (int i = 0; i < A_rows * B_cols; ++i) {
        C[i] = 0.0;
    }
    
    // Sparse matrix-matrix multiplication: C = A * B
    for (int row = 0; row < A_rows; ++row) {
        for (int k_idx = A_csrOffsets[row]; k_idx < A_csrOffsets[row + 1]; ++k_idx) {
            int k = A_columns[k_idx];
            double A_val = A_values[k_idx];
            
            for (int col = 0; col < B_cols; ++col) {
                C[row + col * A_rows] += A_val * B[k + col * B_rows];
            }
        }
    }
}

/*  CPU reference SpMM: C = A^T * B (A sparse CSR, B and C dense col-major) */
static void cpu_spmm_csr_transpose(int A_rows, int A_cols, int A_nnz,
                                  const int *A_csrOffsets, const int *A_columns, const double *A_values,
                                  const double *B, int B_rows, int B_cols,
                                  double *C)
{
    // Initialize C to zero
    for (int i = 0; i < A_cols * B_cols; ++i) {
        C[i] = 0.0;
    }
    
    // Sparse matrix-matrix multiplication: C = A^T * B
    for (int row = 0; row < A_rows; ++row) {
        for (int k_idx = A_csrOffsets[row]; k_idx < A_csrOffsets[row + 1]; ++k_idx) {
            int col = A_columns[k_idx];  // This becomes the row in A^T
            double A_val = A_values[k_idx];
            
            for (int b_col = 0; b_col < B_cols; ++b_col) {
                C[col + b_col * A_cols] += A_val * B[row + b_col * B_rows];
            }
        }
    }
}

static int verify_results(const std::vector<double>& hC, const std::vector<double>& hC_ref, int size)
{
    const double tolerance = 1e-10;
    for (int i = 0; i < size; ++i) {
        if (fabs(hC[i] - hC_ref[i]) > tolerance) {
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    /* Initialize RAFT handle */
    raft::handle_t raft_handle;
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    raft_handle.get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, raft_handle.get_stream()));
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    raft_handle.get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, raft_handle.get_stream()));
    cublasSetStream(raft_handle.get_cublas_handle(), raft_handle.get_stream());
    cusparseSetStream(raft_handle.get_cusparse_handle(), raft_handle.get_stream());

    // Setup up RMM memory pool
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());


    /* ---------------------------------------------------------------- */
    /*  Large sparse matrix in CSR format                              */
    const int   A_NUM_ROWS = 1000;
    const int   A_NUM_COLS = 1000;
    const int   A_NNZ      = 50000;

    std::vector<int> hA_csrOffsets(A_NUM_ROWS + 1);
    std::vector<int> hA_columns(A_NNZ);
    std::vector<double> hA_values(A_NNZ);

    // Generate sparse matrix A with ~5 non-zeros per row on average
    srand(42); // For reproducible results
    int nnz_count = 0;
    hA_csrOffsets[0] = 0;
    
    for (int row = 0; row < A_NUM_ROWS; ++row) {
        int nnz_this_row = (rand() % 8) + 1; // 1-8 non-zeros per row
        if (nnz_count + nnz_this_row > A_NNZ) {
            nnz_this_row = A_NNZ - nnz_count;
        }
        
        for (int j = 0; j < nnz_this_row; ++j) {
            hA_columns[nnz_count] = rand() % A_NUM_COLS;
            hA_values[nnz_count] = (double)(rand() % 10) + 1.0; // Values 1-10
            nnz_count++;
        }
        hA_csrOffsets[row + 1] = nnz_count;
        
        if (nnz_count >= A_NNZ) break;
    }

    /* ---------------------------------------------------------------- */
    /*  Dense matrix B — column-major                                   */
    const int   B_NUM_ROWS = A_NUM_COLS;
    const int   B_NUM_COLS = 10;

    std::vector<double> hB_col(B_NUM_ROWS * B_NUM_COLS);
    for (int i = 0; i < B_NUM_ROWS * B_NUM_COLS; ++i) {
        hB_col[i] = (double)(i % 100) / 10.0; // Values 0.0 to 9.9
    }

    /* ---------------------------------------------------------------- */
    /*  Compute reference results using CPU SpMM                       */
    std::vector<double> hC_ref(A_NUM_ROWS * B_NUM_COLS);
    std::vector<double> hC_AT_ref(A_NUM_COLS * B_NUM_COLS);
    
    cpu_spmm_csr(A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                hA_csrOffsets.data(), hA_columns.data(), hA_values.data(),
                hB_col.data(), B_NUM_ROWS, B_NUM_COLS,
                hC_ref.data());
                
    cpu_spmm_csr_transpose(A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                          hA_csrOffsets.data(), hA_columns.data(), hA_values.data(),
                          hB_col.data(), B_NUM_ROWS, B_NUM_COLS,
                          hC_AT_ref.data());

    std::vector<double> hC(A_NUM_ROWS * B_NUM_COLS);
    std::vector<double> hC_AT(A_NUM_COLS * B_NUM_COLS);
    int overall_ok = 1;

    /* ---------------- variant 1 :  COL / COL ------------------------ */
    float time1 = spmm_col_col(hB_col.data(), hC.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                 hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC, hC_ref, A_NUM_ROWS * B_NUM_COLS)) {
        printf("Variant 1  (B=COL, C=COL) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 2 :  ROW / ROW ------------------------ */
    float time2 = spmm_row_row(hB_col.data(), hC.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                 hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC, hC_ref, A_NUM_ROWS * B_NUM_COLS)) {
        printf("Variant 2  (B=ROW, C=ROW → transpose C) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 3 :  ROW / COL ------------------------ */
    float time3 = spmm_rowcol(hB_col.data(), hC.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC, hC_ref, A_NUM_ROWS * B_NUM_COLS)) {
        printf("Variant 3  (B=ROW -> tranpose B, C=COL) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 4 :  COL / ROW ------------------------ */
    float time4 = spmm_col_row(hB_col.data(), hC.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                 hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC, hC_ref, A_NUM_ROWS * B_NUM_COLS)) {
        printf("Variant 4  (B=COL, C=ROW → transpose C) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 5 :  A^T COL / COL -------------------- */
    float time5 = spmm_AT_col_col(hB_col.data(), hC_AT.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC_AT, hC_AT_ref, A_NUM_COLS * B_NUM_COLS)) {
        printf("Variant 5  (A^T, B=COL, C=COL) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 6 :  A^T ROW / ROW -------------------- */
    float time6 = spmm_AT_row_row(hB_col.data(), hC_AT.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC_AT, hC_AT_ref, A_NUM_COLS * B_NUM_COLS)) {
        printf("Variant 6  (A^T, B=ROW, C=ROW → transpose C) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 7 :  A^T ROW / COL -------------------- */
    float time7 = spmm_AT_rowcol(hB_col.data(), hC_AT.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                   hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC_AT, hC_AT_ref, A_NUM_COLS * B_NUM_COLS)) {
        printf("Variant 7  (A^T, B=ROW → transpose B, C=COL) FAILED\n");
        overall_ok = 0;
    }

    /* ---------------- variant 8 :  A^T COL / ROW -------------------- */
    float time8 = spmm_AT_col_row(hB_col.data(), hC_AT.data(), A_NUM_ROWS, A_NUM_COLS, A_NNZ,
                    hA_csrOffsets.data(), hA_columns.data(), hA_values.data(), B_NUM_ROWS, B_NUM_COLS, &raft_handle);
    if (!verify_results(hC_AT, hC_AT_ref, A_NUM_COLS * B_NUM_COLS)) {
        printf("Variant 8  (A^T, B=COL, C=ROW → transpose C) FAILED\n");
        overall_ok = 0;
    }

    printf("\nOverall test %s\n", overall_ok ? "PASSED" : "FAILED");
    printf("Variant 1  (B=COL, C=COL): %.3f ms\n", time1);
    printf("Variant 2  (B=ROW, C=ROW → transpose C): %.3f ms\n", time2);
    printf("Variant 3  (B=ROW -> tranpose B, C=COL): %.3f ms\n", time3);
    printf("Variant 4  (B=COL, C=ROW → transpose C): %.3f ms\n", time4);
    printf("Variant 5  (A^T, B=COL, C=COL): %.3f ms\n", time5);
    printf("Variant 6  (A^T, B=ROW, C=ROW → transpose C): %.3f ms\n", time6);
    printf("Variant 7  (A^T, B=ROW → transpose B, C=COL): %.3f ms\n", time7);
    printf("Variant 8  (A^T, B=COL, C=ROW → transpose C): %.3f ms\n", time8);

    return overall_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
