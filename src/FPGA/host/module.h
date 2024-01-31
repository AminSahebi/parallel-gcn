#ifndef MODULE_H
#define MODULE_H

#include <immintrin.h>
#include <CL/cl_ext_xilinx.h>
#include "variable.h"
#include "sparse.h"
//#include <CL/cl2.hpp>


class Module {
	public:
		virtual void forward(bool) = 0;
		virtual void backward() = 0;
		virtual ~Module() {};
};

/*class Matmul: public Module {
  Variable *a, *b, *c;
  int m, n, p;
  public:
  Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p);
  ~Matmul() {}
  void forward(bool);
  void backward();
  };
  */
class SparseMatmul: public Module {
	Variable *a, *b, *c;
	SparseIndex *sp;
	int m, n, p;
	public:
	SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p);
	~SparseMatmul() {}
	void forward(bool);
	void backward();
};

class GraphSum: public Module {
	Variable *in, *out;
	SparseIndex *graph;
	int dim;
	public:
	GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim);
	~GraphSum() {}
	void forward(bool);
	void backward();
};

class CrossEntropyLoss: public Module {
	Variable *logits;
	int *truth;
	float *loss;
	int num_classes;
	public:
	CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes);
	~CrossEntropyLoss() {}
	void forward(bool);
	void backward();
};

class ReLU: public Module {
	Variable *in;
	bool *mask;
	public:
	ReLU(Variable *in);
	~ReLU();
	void forward(bool);
	void backward();
};

class Dropout: public Module {
	Variable *in;
	int *mask;
	float p;
	public:
	Dropout(Variable *in, float p);
	~Dropout();
	void forward(bool);
	void backward();
};

class Matmul : public Module {
	Variable *a, *b, *c;
	int m, n, p;
	//      cl_int err;              // Declare here
	//   unsigned fileBufSize;    // Declare here
	//cl::Context context;  // Add context as a member variable

	//cl::CommandQueue queue;  // Declare queue as a member
	cl_command_queue queue = nullptr;

	//cl_kernel kernel;
	cl_kernel krnl;
	cl_mem bufferA, bufferB, bufferC;

	public:
	Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p);

	~Matmul() {}
	void forward(bool training);
	void backward();
};





#endif
