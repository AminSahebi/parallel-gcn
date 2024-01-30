#include <hls_stream.h>

#define BLOCK_SIZE 16
extern "C" {

	void block_mmul(const float A[BLOCK_SIZE][BLOCK_SIZE], const float B[BLOCK_SIZE][BLOCK_SIZE], float C[BLOCK_SIZE][BLOCK_SIZE]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=return bundle=control

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
				C_local[i][j] = C[i][j];
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
