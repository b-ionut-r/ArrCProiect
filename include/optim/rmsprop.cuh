//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_RMSPROP_H
#define ARRC_RMSPROP_H

#include <iostream>
#include "optimizer.h"
#include "ndarray.cuh"

class RMSProp: public Optimizer {
private:
    float beta;
    double eps;
    std::vector<NDArray<float>*> momentum;
public:
    RMSProp(std::vector<tensor::TensorPtrVariant> params, const float &lr,
            const float &weightDecay, const float &beta,
            const double &eps = 1e-8, const ComputeDType &dtype = FLOAT);
    ~RMSProp() override;
    void step() override;
    friend std::ostream & operator<<(std::ostream &os, const RMSProp &rms);
    float getBeta() const {return beta;}
    double getEps() const {return eps;}
};

#endif //ARRC_RMSPROP_H