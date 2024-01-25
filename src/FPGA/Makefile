CURRENT_DIR = $(PWD)
VPP := $(XILINX_VITIS)/bin/v++
EMCONFIGUTIL := $(XILINX_VITIS)/bin/emconfigutil
EMCONFIG_FILE := emconfig.json
#
MODE := hw
#MODE := hw_emu

#PLATFORM := xilinx_u250_gen3x16_xdma_3_1_202020_1
PLATFORM := xilinx_u250_gen3x16_xdma_4_1_202210_1
#PLATFORM := xilinx_u280_gen3x16_xdma_1_202211_1
#PLATFORM := xilinx_u280_xdma_201920_3


#Compiler flags
VPP_COMMON_OPTS := -s -t $(MODE) --platform $(PLATFORM) #  -g --profile.data all:all:all

CFLAGS := -std=c++14 -I$(XILINX_XRT)/include -I$(CURRENT_DIR)/utils #-I/$(CURRENT_DIR)/fpga
#CLAGS += -g

VPP_COMMON_OPTS += -I$(CURRENT_DIR)/kernel --hls.jobs 8
VPP_COMMON_OPTS += --vivado.synth.jobs 8 --vivado.impl.jobs 8 #--advanced.param compiler.axiDeadLockFree=yes -O0
#VPP_COMMON_OPTS += --optimize 3

LFLAGS := -L$(XILINX_XRT)/lib -lxilinxopencl -lpthread -lrt

gcn_app := gcn

# kernel targets
GCN_HW := gcn_kernel_0
GCN_XCLBIN := gcn_kernel_0.xclbin

# kernel rules
GCN_XOS := gcn_kernel_0.xo
GCN_XCLBIN := gcn_kernel_0.xclbin
GCN_CONN := --config $(CURRENT_DIR)/kernel/gcn/connectivity_gcn.cfg #--kernel_frequency=250
gcn_hw_srcdir:= $(CURRENT_DIR)/kernel/gcn
gcn_hw_bindir:= $(CURRENT_DIR)/kernel/gcn/hardware
gcn_hw_src:= $(wildcard $(CURRENT_DIR)/kernel/gcn/gcn_kernel.cpp)

gcn_hw: $(GCN_XCLBIN)

$(gcn_hw_srcdir)/$(GCN_XOS): $(gcn_hw_src)
	$(VPP) $(VPP_COMMON_OPTS) -c -k $(GCN_HW) -o $@ $+

$(GCN_XCLBIN): $(gcn_hw_srcdir)/$(GCN_XOS)
	mkdir -p $(gcn_hw_bindir)
	$(VPP) $(VPP_COMMON_OPTS) -l -o $(gcn_hw_bindir)/$@ $+ $(GCN_CONN) 


gcn_srcdir := $(CURRENT_DIR)/host
gcn_bindir := $(CURRENT_DIR)/host/bin
gcn_src := $(wildcard $(gcn_srcdir)/*.cpp)
gcn_objects := $(patsubst $(gcn_srcdir)/%.cpp, $(gcn_srcdir)/%.o, $(gcn_src))

#.DEFAULT_GOAL := build
#build: $(gcn_bindir)/$(gcn_app)
#
#$(gcn_app): $(gcn_objs)
#	mkdir -p $(gcn_bindir)
#	g++ $(CFLAGS) -o $@ $^ $(LFLAGS)
#
#$(gcn_srcdir)/%.o: $(gcn_srcdir)/%.cpp
#	g++ -c -o $@ $< $(CFLAGS)

.DEFAULT_GOAL := build

build: $(gcn_app)

$(gcn_app): $(gcn_objects)
	mkdir -p $(gcn_bindir)
	g++ $(CFLAGS) -o $@ $^ $(LFLAGS)

$(gcn_srcdir)/%.o: $(gcn_srcdir)/%.cpp
	g++ -c -o $@ $<


run: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	XCL_EMULATION_MODE=${TARGET} ./$(HOST_EXE) $(EXE_OPT)
else
	./$(HOST_EXE) $(EXE_OPT)
endif

# primary build targets
.PHONY: gcn all clean 

clean:
	rm -rf $(EMCONFIG_FILE) $(app) _x .ipcache .run

$(EMCONFIG_FILE):
	$(EMCONFIGUTIL) --nd 1 --od . --platform $(DSA)

check: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
	XCL_EMULATION_MODE=${MODE} ./$(app) $(EXE_OPT)
