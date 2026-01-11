//
// Created by Bujor Ionut Raul on 11.01.2026.
//

#ifndef ARRC_TYPE_TRAITS_CUH
#define ARRC_TYPE_TRAITS_CUH

#include <cuda/std/type_traits>


/// CROSS-TYPE OPERATORS
#define NDARRAY_BINARY_CROSS_OP(OP) \
template<typename D1, typename D2> \
auto operator OP(const NDArray<D1> &a, const NDArray<D2> &b){ \
    using DOut = cuda::std::common_type_t<D1, D2>; \
    return a.template cast<DOut>() OP b.template cast<DOut>(); \
}

#define TENSOR_BINARY_CROSS_OP(OP) \
template<typename D1, typename D2> \
auto operator OP(const Tensor<D1> &a, const Tensor<D2> &b){ \
    using DOut = cuda::std::common_type_t<D1, D2>; \
    return a.template cast<DOut>() OP b.template cast<DOut>(); \
}


#endif //ARRC_TYPE_TRAITS_CUH