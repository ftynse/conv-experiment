LLVM_SRC=$(HOME)/llvm-project
LLVM_BUILD=$(HOME)/llvm-project/build

LLC=$(LLVM_BUILD)/bin/llc
OPT=$(LLVM_BUILD)/bin/opt
MLIR_OPT=$(LLVM_BUILD)/bin/mlir-opt
MLIR_TR=$(LLVM_BUILD)/bin/mlir-translate
CLANGXX=clang
CLANG=$(LLVM_BUILD)/bin/clang
INCLUDE=$(LLVM_SRC)/mlir/include

.PHONY: all
all: conv

conv.td.mlir: conv.mlir
	$(MLIR_OPT) $< --test-transform-dialect-interpreter   --canonicalize --cse --loop-invariant-code-motion > $@

conv.vect.mlir: conv.td.mlir vect.mlir
	# $(MLIR_OPT) $< --linalg-fold-unit-extent-dims --test-transform-dialect-interpreter="transform-file-name=vect.mlir" --canonicalize --cse  --canonicalize > $@
	$(MLIR_OPT) $< --test-transform-dialect-interpreter="transform-file-name=vect.mlir" --canonicalize --cse  --canonicalize > $@

conv.buf.mlir: conv.vect.mlir lower.mlir
	$(MLIR_OPT) $< --test-transform-dialect-interpreter="transform-file-name=lower.mlir" --fold-memref-alias-ops --canonicalize --cse --canonicalize --test-vector-transferop-opt  > $@

# conv.buf.mlir: conv.td.mlir
# 	$(MLIR_OPT) $< --eliminate-empty-tensors --empty-tensor-to-alloc-tensor  -one-shot-bufferize='allow-return-allocs bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' --canonicalize --cse --loop-invariant-code-motion > $@

conv.transformed.mlir: conv.buf.mlir
	$(MLIR_OPT) $< -test-transform-dialect-erase-schedule -test-lower-to-llvm  > $@

conv.ll: conv.transformed.mlir
	$(MLIR_TR) $< --mlir-to-llvmir > $@

# conv.opt.ll: conv.ll
# 	$(OPT) $< -S -o $@ -O3

# conv.s: conv.opt.ll
# 	$(LLC) $< -O3 -o $@

conv.o: conv.ll
	$(CLANG) -x ir -c $< -O3 -march=native -mfma -o $@

# conv: driver.cc conv.s
# 	$(CLANGXX) -std=c++17 $^ -I$(INCLUDE)  $(INCLUDE)/../lib/ExecutionEngine/CRunnerUtils.cpp -O3 -march=native -o $@

conv.s: conv.ll
	$(CLANG) -x ir -c $< -O3 -march=native -mfma -S -o $@

driver.o: driver.cc
	$(CLANG) -x c++ -std=c++17 -c $^ -I$(INCLUDE) -O3 -march=native -o $@

utils.o:
	$(CLANG) -x c++ -std=c++17 -c   $(INCLUDE)/../lib/ExecutionEngine/CRunnerUtils.cpp -I$(INCLUDE) -O3 -march=native -o $@

conv: driver.o utils.o conv.o
	$(CLANG) -O3 -march=native -lstdc++ $^ -o $@

clean:
	rm -f conv.transformed.mlir conv.ll conv.opt.ll conv.s conv.td.mlir
	rm *.o
	rm -f conv
