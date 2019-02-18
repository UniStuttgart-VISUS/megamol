#pragma once

#include <unordered_map>
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathtypes.h"


namespace megamol {
namespace thermodyn {

class PathPCA : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathPCA"; }

    /** Return module class description */
    static const char* Description(void) { return "Compute PCA of particle pathlines"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathPCA();

    virtual ~PathPCA();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    Eigen::MatrixXf computeModes(Eigen::MatrixXf const& x) const;

    /** input of particle pathlines */
    core::CallerSlot dataInSlot_;

    /** output of a subset of particle pathlines */
    core::CalleeSlot dataOutSlot_;

    core::param::ParamSlot numFeatureSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    std::vector<std::vector<float>> par_data_;

    std::vector<float> minV_;
    std::vector<float> maxV_;

    std::vector<int> entrySizes_;

    std::vector<bool> colsPresent_;

    std::vector<bool> dirsPresent_;

    std::vector<std::unordered_map<uint64_t, std::vector<float>>> pathStore_;

    vislib::math::Cuboid<float> bbox_;

    using svdsolver = Eigen::BDCSVD<Eigen::MatrixXf>;

    using eigensolver = Eigen::EigenSolver<Eigen::MatrixXf>;
};

} // end namespace thermodyn
} // end namespace megamol