#ifndef ARRC_FUNCTIONS_BASE_H
#define ARRC_FUNCTIONS_BASE_H

#include <memory>
#include <utility>
#include <vector>
#include "ndarray.cuh"

template <typename dtype>
struct TensorImpl;

namespace functions {

template <typename dtype>
class Function {
protected:
    std::vector<std::shared_ptr<TensorImpl<dtype>>> parents;
public:
    explicit Function(std::vector<std::shared_ptr<TensorImpl<dtype>>> parents)
        : parents(std::move(parents)) {}
    virtual ~Function() = default;
    const std::vector<std::shared_ptr<TensorImpl<dtype>>> &getParents() const {
        return parents;
    }
    virtual void backward(const NDArray<dtype> &gradOutput) = 0;
};

} // namespace functions

#endif // ARRC_FUNCTIONS_BASE_H
