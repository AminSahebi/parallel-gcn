#include <fstream>
#include <stdexcept>
#include <iostream>
#include <array>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>
#include <numeric>
#include <cstring>
//#define DEBUG1
//#include "module.h"
#include "optim.h"
#include "variable.h"
#include "parser.h"
#include <CL/cl_ext_xilinx.h>

#include "globals.h"

#define DATA_SIZE 64 * 4096 
#define num_cu  15

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 210
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

//OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error,call)                                      		 \
	call;                                                           	 \
	if (error != CL_SUCCESS) {                                      	 \
		printf("%s:%d Error calling " #call ", error code is: %d\n",  	 \
				__FILE__,__LINE__, error);                       \
		exit(EXIT_FAILURE);                                           	 \
	}

using namespace std;

template <typename T>
struct aligned_allocator
{
	using value_type = T;
	T* allocate(std::size_t num)
	{
		void* ptr = nullptr;
		if (posix_memalign(&ptr,4096,num*sizeof(T)))
			throw std::bad_alloc();
		return reinterpret_cast<T*>(ptr);
	}
	void deallocate(T* p, std::size_t num)
	{
		free(p);
	}
};




int main(int argc, char** argv) {
	unsigned fileBufSize;

	if (argc < 3) {
		cout << "parallel_gcn graph_name bitsream [num_nodes input_dim hidden_dim output_dim"
			<< "dropout learning_rate, weight_decay epochs early_stopping]" << endl;
		return EXIT_FAILURE;
	}

	std::string binaryFile = argv[2];

	// Load graph data and GCN parameters
	GCNParams params = GCNParams::get_default();
	GCNData data;
	string input_name(argv[1]);
	Parser parser(&params, &data, input_name);
	if (!parser.parse()) {
		cerr << "Cannot read input: " << input_name << endl;
		exit(EXIT_FAILURE);
	}
	fnm = argv[2];



	GCN gcn(params, &data);


	gcn.run(); // to be deployed on kernels
	return EXIT_SUCCESS;



}
