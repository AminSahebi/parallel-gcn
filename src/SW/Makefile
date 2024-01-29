UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	CXX=g++
endif
ifeq ($(UNAME), Darwin)
        CXX=gcc
endif
#CXX=gcc
CXXFLAGS= -O3 -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++
CXXFILES=gcn.cpp optim.cpp module.cpp variable.cpp parser.cpp rand.cpp timer.cpp
SIMD_CXXFILES=simdxorshift128plus.cpp xorshift128plus.cpp
SIMD_HFILES=xorshift128plus.h simdaesdragontamer.h simdxorshift128plus.h 
HFILES=gcn.h optim.h module.h variable.h sparse.h parser.h rand.h timer.h 
TEST_CXXFILES=test/module_test.cpp test/optim_test.cpp test/util.cpp
TEST_HFILES=test/util.h
OMP=-fopenmp -DOMP

#SIMD=-DSIMD
#SIMD=-mavx -march=native #-DSIMD
SIMD=-mavx2 -march=native -DSIMD
all: seq omp omp-simd

seq: main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) main.cpp $(LDFLAGS)

omp-simd: main.cpp $(CXXFILES) $(HFILES) $(SIMD_HFILES)
	$(CXX) $(CXXFLAGS) $(OMP) $(SIMD) -o gcn-omp-simd $(SIMD_CXXFILES) $(CXXFILES) main.cpp $(LDFLAGS)

omp: main.cpp $(CXXFILES) $(HFILES)
	$(CXX) -fno-tree-vectorize $(CXXFLAGS) $(OMP) -o gcn-omp $(CXXFILES) main.cpp $(LDFLAGS)




test: $(CXXFILES) $(HFILES) $(TEST_CXXFILES) $(TEST_HFILES)
	$(CXX) $(CXXFLAGS) -Iinclude -o gcn-test $(CXXFILES) $(TEST_CXXFILES) test/main.cpp $(LDFLAGS)
