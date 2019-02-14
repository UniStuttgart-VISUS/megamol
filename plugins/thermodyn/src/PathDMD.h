#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "Eigen/SVD"
#include "Eigen/Eigenvalues"


namespace megamol {
namespace thermodyn {

class PathDMD : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathDMD"; }

    /** Return module class description */
    static const char* Description(void) { return "Compute DMD of particle pathlines"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathDMD();

    virtual ~PathDMD();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    /** input of particle pathlines */
    core::CallerSlot dataInSlot_;

    /** output of a subset of particle pathlines */
    core::CalleeSlot dataOutSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    std::vector<std::vector<float>> par_data_;

    std::vector<float> minV_;
    std::vector<float> maxV_;

    using svdsolver = Eigen::BDCSVD<Eigen::MatrixXf>;

    using eigensolver = Eigen::EigenSolver<Eigen::MatrixXf>;

};

} // end namespace thermodyn
} // end namespace megamol