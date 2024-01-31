#include "module.h"
#include "rand.h"
#include "timer.h"
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include "globals.h"
#include <fstream>
//#include <CL/cl2.hpp>


#ifdef OMP
#include <omp.h>
#endif


#define BLOCK_SIZE 16
	static void
throw_if_error(cl_int errcode, const char* msg=nullptr)
{
	if (!errcode)
		return;
	std::string err = "errcode '";
	err.append(std::to_string(errcode)).append("'");
	if (msg)
		err.append(" ").append(msg);
	throw std::runtime_error(err);
}


Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p)
	: a(a), b(b), c(c), m(m), n(n), p(p) { 

		//program the FPGA 
		cl_int err = CL_SUCCESS;
		cl_int status = CL_SUCCESS;

		cl_platform_id platform = nullptr;
		throw_if_error(clGetPlatformIDs(1,&platform,nullptr));

		cl_uint num_devices = 0;
		throw_if_error(clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,0,nullptr,&num_devices));
		throw_if_error(num_devices==0,"no devices");

		std::vector<cl_device_id> devices(num_devices);

		throw_if_error(clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,num_devices,devices.data(),nullptr));

		cl_device_id device = devices.front();

		cl_context context = clCreateContext(0,1,&device,nullptr,nullptr,&err);

		throw_if_error(err);

		queue = clCreateCommandQueue(context,device,CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,&err);

		throw_if_error(err,"failed to create command queue");


		std::ifstream stream(fnm);

		stream.seekg(0,stream.end);
		size_t size = stream.tellg();

		stream.seekg(0,stream.beg);
		std::vector<char> xclbin(size);
		stream.read(xclbin.data(),size);

		const unsigned char* Data = reinterpret_cast<unsigned char*>(xclbin.data());
		cl_program program = clCreateProgramWithBinary(context,1,&device,&size,&Data,&status,&err);
		throw_if_error(err,"failed to create program");
		printf("FPGA program done successfully!\n");

		//Create Kernels
		krnl = clCreateKernel(program,"mmul_kernel_0",&err);
		throw_if_error(err,"failed to allocate kernel");	

		bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(float) * m * n, a->data.data(),&err);
		bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(float) * n * p, b->data.data(),&err);
		bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(float) * m * p, c->data.data(),&err);

		//		clSetKernelArg(krnl, 0, sizeof(cl_mem) ,&bufferA);
		//		clSetKernelArg(krnl, 1, sizeof(cl_mem) ,&bufferB);
		//		clSetKernelArg(krnl, 2, sizeof(cl_mem) ,&bufferC);
	}

void Matmul::forward(bool training) {
	timer_start(TMR_MATMUL_FW);
	cl_event write_event;
	cl_event ev_kernel_done;
	cl_event read_done;

	cl_mem mems[3] = {bufferA,bufferB, bufferC};
	clEnqueueMigrateMemObjects(queue,3,mems,0,0,nullptr,&write_event);


	//	clEnqueueTask(queue,krnl,1,&write_event,&ev_kernel_done);
	for (int bi = 0; bi < m; bi += BLOCK_SIZE) {
		for (int bj = 0; bj < p; bj += BLOCK_SIZE) {
			for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
				// Calculate actual block sizes for this iteration
				int current_block_size_i = std::min(BLOCK_SIZE, m - bi);
				int current_block_size_j = std::min(BLOCK_SIZE, p - bj);
				int current_block_size_k = std::min(BLOCK_SIZE, n - bk);

				// Set kernel arguments
				clSetKernelArg(krnl, 0, sizeof(cl_mem), &bufferA);
				clSetKernelArg(krnl, 1, sizeof(cl_mem), &bufferB);
				clSetKernelArg(krnl, 2, sizeof(cl_mem), &bufferC);
				clSetKernelArg(krnl, 3, sizeof(int), &bi);  // Pass block indices as additional arguments
				clSetKernelArg(krnl, 4, sizeof(int), &bj);
				clSetKernelArg(krnl, 5, sizeof(int), &bk);
				clSetKernelArg(krnl, 6, sizeof(int), &current_block_size_i);
				clSetKernelArg(krnl, 7, sizeof(int), &current_block_size_j);
				clSetKernelArg(krnl, 8, sizeof(int), &current_block_size_k);

				// Enqueue the kernel for each block
				size_t global_size[2] = {static_cast<size_t>(current_block_size_i), static_cast<size_t>(current_block_size_j)};
				size_t local_size[2] = {1, 1};
				clEnqueueNDRangeKernel(queue, krnl, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
			}
		}
	}

	clEnqueueMigrateMemObjects(queue,1,&bufferC,CL_MIGRATE_MEM_OBJECT_HOST,1,&ev_kernel_done,&read_done);

	clReleaseEvent(write_event);
	clReleaseEvent(ev_kernel_done);
	timer_stop(TMR_MATMUL_FW);
}





/*
   Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) :
   a(a), b(b), c(c), m(m), n(n), p(p) {}

   void Matmul::forward(bool training) {

#ifdef DEBUG
printf("Matmul forward is called\n");
#endif
timer_start(TMR_MATMUL_FW);
c->zero();
#pragma omp parallel for schedule(static)
for (int i = 0; i < m; i++)
for (int j = 0; j < n; j++) {
#ifdef SIMD
#pragma omp simd
#endif
for (int k = 0; k < p; k++)
c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
}
timer_stop(TMR_MATMUL_FW);
}
*/
void Matmul::backward() {
#ifdef DEBUG
	printf("Matmul backward is called\n");
#endif
	timer_start(TMR_MATMUL_BW);
	a->zero_grad();
	b->zero_grad();
#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			float tmp = 0;
#ifdef SIMD
#pragma omp simd reduction(+:tmp)
#endif
			for (int k = 0; k < p; k++) {
				tmp += c->grad[i * p + k] * b->data[j * p + k];
#ifdef OMP
				b->local_grad[omp_get_thread_num()][j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
#else
				b->grad[j * p + k] += c->grad[i * p + k] * a->data[i * n + j];
#endif
			}
			a->grad[i * n + j] = tmp;
		}
#ifdef OMP
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
	for(int i = 0; i < b->grad.size(); i++)
		for(int thread = 0; thread < omp_get_num_threads(); thread++)
			b->grad[i] += b->local_grad[thread][i];
#endif
	timer_stop(TMR_MATMUL_BW);
}

SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) :
	a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

	void SparseMatmul::forward(bool training) {
#ifdef DEBUG
		printf("Sparse Matmul forward is called\n");
#endif
		timer_start(TMR_SPMATMUL_FW);
		c->zero();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < sp->indptr.size() - 1; i++)
			for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
				int j = sp->indices[jj];
#ifdef SIMD
#pragma omp simd
#endif
				for (int k = 0; k < p; k++)
					c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
			}
		timer_stop(TMR_SPMATMUL_FW);
	}

void SparseMatmul::backward() {
#ifdef DEBUG
	printf("Sparse Matmul backward is called\n");
#endif
	timer_start(TMR_SPMATMUL_BW);
	b->zero_grad();
	int row = 0;
#pragma omp parallel for schedule(static)
	for (int i = 0; i < sp->indptr.size() - 1; i++)
		for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
			int j = sp->indices[jj];
#ifdef SIMD
#pragma omp simd
#endif
			for (int k = 0; k < p; k++)
#ifdef OMP
				b->local_grad[omp_get_thread_num()][j * p + k] += c->grad[i * p + k] * a->data[jj];
#else
			b->grad[j * p + k] += c->grad[i * p + k] * a->data[jj];
#endif
		}
#ifdef OMP
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
	for(int i = 0; i < b->grad.size(); i++)
		for(int thread = 0; thread < omp_get_num_threads(); thread++)
			b->grad[i] += b->local_grad[thread][i];
#endif
	timer_stop(TMR_SPMATMUL_BW);
}

GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) :
	in(in), out(out), graph(graph), dim(dim) {}

	void GraphSum::forward(bool training) {

#ifdef DEBUG
		printf("Graphsum forward is called\n");
#endif
		timer_start(TMR_GRAPHSUM_FW);
		out->zero();
#pragma omp parallel for schedule(static)
		for (int src = 0; src < graph->indptr.size() - 1; src++)
			for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
				int dst = graph->indices[i];
				float coef = 1.0 / sqrtf(
						(graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
						);
#ifdef SIMD
#pragma omp simd
#endif
				for (int j = 0; j < dim; j++)
					// This only works for undirected graphs. Should be out[dst] += coef * in[src]
					out->data[src * dim + j] += coef * in->data[dst * dim + j];
			}
		timer_stop(TMR_GRAPHSUM_FW);
	}

void GraphSum::backward() {
#ifdef DEBUG
	printf("Graphsum backward is called\n");
#endif
	timer_start(TMR_GRAPHSUM_BW);
	in->zero_grad();
#pragma omp parallel for schedule(static)
	for (int src = 0; src < graph->indptr.size() - 1; src++)
		for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
			int dst = graph->indices[i];
			float coef = 1.0 / sqrtf(
					(graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
					);
#ifdef SIMD
#pragma omp simd
#endif
			for (int j = 0; j < dim; j++)
				in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
		}
	timer_stop(TMR_GRAPHSUM_BW);
}

CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) :
	logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

	void CrossEntropyLoss::forward(bool training) {
		timer_start(TMR_LOSS_FW);
		float total_loss = 0;
		int count = 0;
		if (training) logits->zero_grad();
#pragma omp parallel for schedule(static) reduction(+:total_loss) reduction(+:count)
		for (int i = 0; i < logits->data.size() / num_classes; i++) {
			if (truth[i] < 0) continue;
			count++;
			float *logit = &logits->data[i * num_classes];
			float max_logit = -1e30, sum_exp = 0;
#ifdef SIMD
#pragma omp simd reduction(max:max_logit)
#endif
			for (int j = 0; j < num_classes; j++)
				max_logit = fmax(max_logit, logit[j]);
#ifdef SIMD
#pragma omp simd reduction(+:sum_exp)
#endif
			for (int j = 0; j < num_classes; j++) {
				logit[j] -= max_logit;
				sum_exp += expf(logit[j]);
			}
			total_loss += logf(sum_exp) - logit[truth[i]];

			if (training) {
#ifdef SIMD
#pragma omp simd
#endif
				for (int j = 0; j < num_classes; j++) {
					float prob = expf(logit[j]) / sum_exp;
					logits->grad[i * num_classes + j] = prob;
				}
				logits->grad[i * num_classes + truth[i]] -= 1.0;
			}
		}
		*loss = total_loss / count;
		if (training)
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
			for (int i = 0; i < logits->grad.size(); i++)
				logits->grad[i] /= count;
		timer_stop(TMR_LOSS_FW);
	}

void CrossEntropyLoss::backward() {
}

ReLU::ReLU(Variable *in) {
	this->in = in;
	mask = new bool[in->data.size()];
}

ReLU::~ReLU() {
	delete[] mask;
}

void ReLU::forward(bool training) {
	timer_start(TMR_RELU_FW);
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < in->data.size(); i++) {
		bool keep = in->data[i] > 0;
		if (training) mask[i] = keep;
		if (!keep) in->data[i] = 0;
	}
	timer_stop(TMR_RELU_FW);
}

void ReLU::backward() {
	timer_start(TMR_RELU_BW);
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < in->data.size(); i++)
		if (!mask[i]) in->grad[i] = 0;
	timer_stop(TMR_RELU_BW);
}

Dropout::Dropout(Variable *in, float p) {
	this->in = in;
	this->p = p;
	if (!in->grad.empty()) mask = new int[in->data.size()];
	else mask = nullptr;
}

Dropout::~Dropout() {
	if (mask) delete[] mask;
}

void Dropout::forward(bool training) {
	if (!training) return;
	timer_start(TMR_DROPOUT_FW);
	const int threshold = int(p * MY_RAND_MAX);
	float scale = 1 / (1 - p);

#ifdef SIMD
#pragma omp parallel for schedule(static)
	for (int i = 0; i < in->data.size() / 8; i++) {
		__m256i rand_v = gen_simd_rand(omp_get_thread_num());
		__m256i threshold_v = _mm256_set1_epi32(threshold);
		__m256 data = _mm256_loadu_ps(&in->data[8 * i]);
		__m256i mask1 = _mm256_cmpgt_epi32(rand_v, threshold_v);
		__m256 scale_v = _mm256_blendv_ps(_mm256_set1_ps(scale), _mm256_setzero_ps(), (__m256) mask1);
		data = _mm256_mul_ps(data, scale_v);
		_mm256_storeu_ps(&in->data[8 * i], data);
		if (mask) {
			_mm256_storeu_si256( (__m256i *)&mask[8 * i], mask1);
		}
	}
#else
#pragma omp parallel for schedule(static)
	for (int i = 0; i < in->data.size(); i++) {
		bool keep = (int)RAND() >= threshold;
		in->data[i] *= keep ? scale : 0;
		if (mask) mask[i] = keep;
	}
#endif
	timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
	if (!mask) return;
	timer_start(TMR_DROPOUT_BW);
	float scale = 1 / (1 - p);
#ifdef SIMD
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < in->data.size(); i++)
		in->grad[i] *= mask[i] ? scale : 0;
	timer_stop(TMR_DROPOUT_BW);
}
