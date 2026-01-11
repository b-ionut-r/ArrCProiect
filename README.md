# ArrC - CUDA Autograd Framework

ArrC is a small CUDA-first deep learning playground built as an OOP project. It provides GPU-backed NDArrays, an autograd Tensor wrapper, and CUDA optimizers, all demonstrated via a CLI menu.

## Requirements

- CMake 3.26+
- C++ compiler with C++23 support
- CUDA Toolkit (nvcc) for GPU builds

## Build and Run

### Windows (Ninja)
```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
.\build\oop.exe
```

### Generic CMake
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

### Optional Scripts (bash)
```sh
./scripts/cmake.sh configure
./scripts/cmake.sh build
```

## Quick Usage

### NDArray
```cpp
#include "ndarray.cuh"

NDArray<float> a({2, 2});
NDArray<float> b({2, 2});
a = 5.0f;
b = 2.0f;

auto c = a + b;
std::cout << c << std::endl;

auto view = a[std::vector<Slice>{Slice(0, 2), Slice(0, 1)}];
```

### Tensor + Autograd
```cpp
#include "tensor.h"
#include "functions/arithmetic.h"

Tensor<float> x = tensor::zeros<float>({1}, true);
Tensor<float> y = tensor::zeros<float>({1}, true);
x.data()[std::vector<int>{0}] = 3.0f;
y.data()[std::vector<int>{0}] = 4.0f;

auto z = x * y;
z.backward();

std::cout << "dz/dx = " << x.grad() << std::endl;
std::cout << "dz/dy = " << y.grad() << std::endl;
```

### Optimizer
```cpp
#include "optim/sgd.cuh"

Tensor<float> w = tensor::ones<float>({2, 2}, true);
w.grad() = 0.1f;

std::vector<tensor::TensorPtrVariant> params = {w.get()};
SGD opt(params, 0.01f, 0.0f, 0.9f);
opt.step();
```

## Architecture Overview

### NDArray<T>

GPU-backed N-dimensional array with:

- Unified Memory allocation (`cudaMallocManaged`) and RAII cleanup
- Shape, strides, offset, and view semantics (slicing is non-owning)
- Elementwise ops (+, -, *, /) with broadcasting support (same ndim required)
- Strided and contiguous CUDA kernels
- Deep copy on copy construction; move transfers ownership; views do not own data
- Global GPU memory tracking via `NDArray<T>::getTotalAllocatedMemory()`

Broadcasting is implemented in the forward pass by creating zero-stride views. Shape mismatches throw `ShapeMismatchException`.

### Tensor<T> and Autograd

`Tensor<T>` is a lightweight handle around a shared `TensorImpl<T>` (copying a
Tensor shares the same underlying storage):

- `TensorImpl` stores `data`, `grad`, `requiresGrad`, `hasGrad`, and a `gradFn`
- `grad()` allocates on first use and accumulates via `accumulateGrad()`
- `backward()` builds a topological order from `Function` parents, then runs reverse
- Ops are overloaded (`+`, `-`, `*`, `/`, unary `-`) and attach `Function` nodes

Autograd design details:

- No global tape. The graph is owned by Tensors through `gradFn`.
- Gradients are accumulated in-place using NDArray ops.
- `Tensor::cast<newDtype>()` attaches a lightweight cast edge:
  - The casted tensor stores callbacks to cast the gradient back to the parent dtype.
  - After accumulation, the parent subgraph backpropagates once.
- Cross-type ops use a common type through `Tensor::cast`.
- `Tensor::get()` returns a non-owning pointer for optimizer parameter lists.

Limitations:

- Gradient reduction for broadcasted shapes is not implemented in
  `functions::detail::reduceGradToShape` and will throw.
- Only elementwise arithmetic has autograd rules (no matmul/conv yet).

### Functions

`functions::Function<T>` is a simple abstract base class storing parent tensors.
Implemented ops in `include/functions/arithmetic.h`:

- `AddFunction`
- `SubFunction`
- `MulFunction`
- `DivFunction`
- `NegFunction`

### Optimizers

Base class: `Optimizer` with a Strategy-style `step()` method.
Concrete optimizers:

- `SGD` (momentum)
- `Adam` (optionally AdamW)
- `RMSProp`

Implementation details:

- Parameters are stored as `tensor::TensorPtrVariant` for type flexibility.
- Momentum buffers are `NDArray<float>` allocated on first construction.
- Compute type is selected by `ComputeDType` (HALF/FLOAT/DOUBLE).
- CUDA kernels are fused and invoked per parameter, with `t` incremented per step.
- Optimizer state is allocated in constructors and released in destructors.

### Exceptions

All errors derive from `ArrcException`:

- `SizeMismatchException`
- `NDimMismatchException`
- `ShapeMismatchException`
- `IndexingException`
- `CudaKernelException`
- `BackPropException`

## Demo Program

`main.cu` provides an interactive menu:

1. NDArray demo (templates, operators, copy)
2. Tensor + autograd demo
3. Optimizers demo (polymorphism, upcast/downcast)
4. Exceptions demo
5. Training demo (minimize `(x - 3)^2`)
6. Framework info

## Project Layout

- `include/ndarray.cuh` - NDArray implementation and kernels interface
- `include/tensor.h` - Tensor + autograd
- `include/functions/*` - Autograd Function nodes
- `include/optim/*` - Optimizer interfaces
- `src/optim/*` - Optimizer CUDA implementations
- `main.cu` - Demo CLI
