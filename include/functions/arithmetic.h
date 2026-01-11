#ifndef ARRC_FUNCTIONS_ARITHMETIC_H
#define ARRC_FUNCTIONS_ARITHMETIC_H

#include <memory>
#include <utility>
#include <vector>
#include "exceptions.h"
#include "tensor.h"
#include "type_traits.cuh"

namespace functions {
namespace detail {

template <typename dtype>
NDArray<dtype> reduceGradToShape(const NDArray<dtype> &grad,
                                 const std::vector<int> &targetShape) {
    const auto gradShape = grad.getShape();
    if (gradShape == targetShape) return grad;
    // TODO: Implement proper broadcasting gradient reduction
    // custom reduction CUDA Kernel may be needed here
    throw ShapeMismatchException("Gradient reduction to target shape not implemented.");
}

} // namespace detail

template <typename dtype>
class AddFunction : public Function<dtype> {
public:
    AddFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}
    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a && a->requiresGrad) {
            a->accumulateGrad(detail::reduceGradToShape(gradOutput, a->data.getShape()));
        }
        if (b && b->requiresGrad) {
            b->accumulateGrad(detail::reduceGradToShape(gradOutput, b->data.getShape()));
        }
    }
};

template <typename dtype>
class SubFunction : public Function<dtype> {
public:
    SubFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}
    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a && a->requiresGrad) {
            a->accumulateGrad(detail::reduceGradToShape(gradOutput, a->data.getShape()));
        }
        if (b && b->requiresGrad) {
            NDArray<dtype> neg = -gradOutput;
            b->accumulateGrad(detail::reduceGradToShape(neg, b->data.getShape()));
        }
    }
};

template <typename dtype>
class MulFunction : public Function<dtype> {
public:
    MulFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}
    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a && a->requiresGrad) {
            NDArray<dtype> gradA = gradOutput * b->data;
            a->accumulateGrad(detail::reduceGradToShape(gradA, a->data.getShape()));
        }
        if (b && b->requiresGrad) {
            NDArray<dtype> gradB = gradOutput * a->data;
            b->accumulateGrad(detail::reduceGradToShape(gradB, b->data.getShape()));
        }
    }
};

template <typename dtype>
class DivFunction : public Function<dtype> {
public:
    DivFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}
    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a && a->requiresGrad) {
            NDArray<dtype> gradA = gradOutput / b->data;
            a->accumulateGrad(detail::reduceGradToShape(gradA, a->data.getShape()));
        }
        if (b && b->requiresGrad) {
            NDArray<dtype> denom = b->data * b->data;
            NDArray<dtype> gradB = gradOutput * a->data / denom;
            gradB = -gradB;
            b->accumulateGrad(detail::reduceGradToShape(gradB, b->data.getShape()));
        }
    }
};

template <typename dtype>
class NegFunction : public Function<dtype> {
public:
    explicit NegFunction(std::shared_ptr<TensorImpl<dtype>> a)
        : Function<dtype>({std::move(a)}) {}
    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        if (a && a->requiresGrad) {
            NDArray<dtype> gradA = -gradOutput;
            a->accumulateGrad(detail::reduceGradToShape(gradA, a->data.getShape()));
        }
    }
};

} // namespace functions

template <typename dtype>
Tensor<dtype> operator+(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    NDArray<dtype> outData = a.data() + b.data();
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) {
        out.setGradFn(std::make_shared<functions::AddFunction<dtype>>(a.getImpl(), b.getImpl()));
    }
    return out;
}

template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    NDArray<dtype> outData = a.data() - b.data();
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) {
        out.setGradFn(std::make_shared<functions::SubFunction<dtype>>(a.getImpl(), b.getImpl()));
    }
    return out;
}

template <typename dtype>
Tensor<dtype> operator*(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    NDArray<dtype> outData = a.data() * b.data();
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) {
        out.setGradFn(std::make_shared<functions::MulFunction<dtype>>(a.getImpl(), b.getImpl()));
    }
    return out;
}

template <typename dtype>
Tensor<dtype> operator/(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    NDArray<dtype> outData = a.data() / b.data();
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) {
        out.setGradFn(std::make_shared<functions::DivFunction<dtype>>(a.getImpl(), b.getImpl()));
    }
    return out;
}

template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype> &a) {
    NDArray<dtype> outData = -a.data();
    bool requiresGrad = a.requiresGrad();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) {
        out.setGradFn(std::make_shared<functions::NegFunction<dtype>>(a.getImpl()));
    }
    return out;
}

TENSOR_BINARY_CROSS_OP(+)
TENSOR_BINARY_CROSS_OP(-)
TENSOR_BINARY_CROSS_OP(*)
TENSOR_BINARY_CROSS_OP(/)

#endif // ARRC_FUNCTIONS_ARITHMETIC_H
