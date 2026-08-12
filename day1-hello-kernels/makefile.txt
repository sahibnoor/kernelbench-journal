# Makefile for hello-kernels
# make          -> builds the CUDA binary
# make run      -> builds + runs both CUDA and Triton
# make clean    -> removes the compiled binary

CC := g++-13
NVCC := nvcc
NVCC_FLAGS := -ccbin $(CC)

CUDA_SRC := vector_add.cu
CUDA_BIN := vector_add
TRITON_SRC := vector_add_triton.py

.PHONY: all run run-cuda run-triton clean

all: $(CUDA_BIN)

# $< = the first dependency (vector_add.cu), $@ = the target name (vector_add)
$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run-cuda: $(CUDA_BIN)
	./$(CUDA_BIN)

run-triton:
	python $(TRITON_SRC)

run: run-cuda run-triton

clean:
	rm -f $(CUDA_BIN)