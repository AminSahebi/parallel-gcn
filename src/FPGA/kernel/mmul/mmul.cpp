/*
#include <hls_stream.h>
#define BLOCK_SIZE 512
extern "C" {

	void mmul_kernel_0(
		       const float A[BLOCK_SIZE][BLOCK_SIZE],
                       const float B[BLOCK_SIZE][BLOCK_SIZE],
                       float C[BLOCK_SIZE][BLOCK_SIZE],
                       int bi, 
		       int bj,
		       int bk,
                       int current_block_size_i, 
		       int current_block_size_j, 
		       int current_block_size_k) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2

        // Matrix multiplication implementation using Vivado HLS pragmas
        float A_local[BLOCK_SIZE][BLOCK_SIZE];
        float B_local[BLOCK_SIZE][BLOCK_SIZE];
        float C_local[BLOCK_SIZE][BLOCK_SIZE];

        // Load data from global memory to local memory
        for (int i = 0; i < current_block_size_i; ++i) {
            for (int j = 0; j < current_block_size_k; ++j) {
#pragma HLS pipeline II=1
                A_local[i][j] = A[i + bi][j + bk];
            }
        }

        for (int i = 0; i < current_block_size_k; ++i) {
            for (int j = 0; j < current_block_size_j; ++j) {
#pragma HLS pipeline II=1
                B_local[i][j] = B[i + bk][j + bj];
            }
        }

        for (int i = 0; i < current_block_size_i; ++i) {
            for (int j = 0; j < current_block_size_j; ++j) {
#pragma HLS pipeline II=1
                C_local[i][j] = C[i + bi][j + bj];
            }
        }

        // Perform matrix multiplication
        for (int i = 0; i < current_block_size_i; ++i) {
            for (int j = 0; j < current_block_size_j; ++j) {
#pragma HLS pipeline II=1
                float sum = 0;
                for (int k = 0; k < current_block_size_k; ++k) {
#pragma HLS unroll
                    sum += A_local[i][k] * B_local[k][j];
                }
                C_local[i][j] += sum;
            }
        }

        // Store the result back to global memory
        for (int i = 0; i < current_block_size_i; ++i) {
            for (int j = 0; j < current_block_size_j; ++j) {
#pragma HLS pipeline II=1
                C[i + bi][j + bj] = C_local[i][j];
            }
        }
    }
}

*/

#include <hls_stream.h>

#define BLOCK_SIZE 64
extern "C" {

	void mmul_kernel_0(const float A[BLOCK_SIZE][BLOCK_SIZE], const float B[BLOCK_SIZE][BLOCK_SIZE], float C[BLOCK_SIZE][BLOCK_SIZE]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2


		// Matrix multiplication implementation using Vivado HLS pragmas
		float A_local[BLOCK_SIZE][BLOCK_SIZE];
		float B_local[BLOCK_SIZE][BLOCK_SIZE];
		float C_local[BLOCK_SIZE][BLOCK_SIZE];

		// Load data from global memory to local memory
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			for (int j = 0; j < BLOCK_SIZE; ++j) {
#pragma HLS pipeline II=1
				A_local[i][j] = A[i][j];
				B_local[i][j] = B[i][j];
		//		C_local[i][j] = C[i][j];
			}
		}

		// Perform matrix multiplication
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			for (int j = 0; j < BLOCK_SIZE; ++j) {
#pragma HLS pipeline II=1
				float sum = 0;
				for (int k = 0; k < BLOCK_SIZE; ++k) {
#pragma HLS unroll
					sum += A_local[i][k] * B_local[k][j];
				}
				C_local[i][j] += sum;
			}
		}

		// Store the result back to global memory
		for (int i = 0; i < BLOCK_SIZE; ++i) {
			for (int j = 0; j < BLOCK_SIZE; ++j) {
#pragma HLS pipeline II=1
				C[i][j] = C_local[i][j];
			}
		}
	}
}

