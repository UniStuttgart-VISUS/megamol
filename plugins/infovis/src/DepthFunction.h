#pragma once

#include "datatools/table/TableDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include <Eigen/Core>

namespace megamol::infovis {

using namespace megamol::core;

class DepthFunction : public core::Module {
public:
    /** Return module class name */
    static inline const char* ClassName() {
        return "DepthFunction";
    }

    /** Return module class description */
    static inline const char* Description() {
        return "Non-parametric depth functions (order statistics) for multivariate analysis";
    }

    /** Module is always available */
    static inline bool IsAvailable() {
        return true;
    }

    static Eigen::VectorXd halfSpaceDepth(Eigen::MatrixXd dataMatrix);
    static Eigen::MatrixXd mahalanobisDepth(Eigen::MatrixXd dataMatrix);
    static Eigen::VectorXd functionalDepth(
        Eigen::MatrixXd dataMatrix, int samplesCount, int samplesLength, unsigned int seed);

    Eigen::VectorXd simplicalDepth(Eigen::MatrixXd dataMatrix, int samplesCount, unsigned int seed);

    /** Constructor */
    DepthFunction();

    /** Destructor */
    ~DepthFunction() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

private:
    /** Data callback */
    bool getDataCallback(core::Call& c);

    bool getHashCallback(core::Call& c);

    void assertData(megamol::datatools::table::TableDataCall* inCall);

    /** check and set parameters dirty/undirty*/
    bool paramsIsDirty();
    void paramsResetDirty();

    /** compute function */
    bool apply(megamol::datatools::table::TableDataCall* inCall);

    /** compute binomial coefficient */
    static double binomialCoeff(int n, int k);

    /** Data output slot */
    CalleeSlot dataOutSlot;

    /** Data output slot */
    CallerSlot dataInSlot;

    /* Column groups */
    ::megamol::core::param::ParamSlot columnGroupsSlot;

    /** Depth function type */
    ::megamol::core::param::ParamSlot depthType;

    /** Number of samples */
    ::megamol::core::param::ParamSlot sampleCount;

    /** Number of samples */
    ::megamol::core::param::ParamSlot sampleLength;

    /** Seed */
    ::megamol::core::param::ParamSlot randomSeed;

    /** Hash of the current data */
    size_t datahash;
    size_t dataInHash;

    /** Vector storing information about columns and rows*/
    std::vector<megamol::datatools::table::TableDataCall::ColumnInfo> columnInfos;

    /** list of all parameters*/
    std::vector<::megamol::core::param::ParamSlot*> params;

    /** Vector stroing the actual float data */
    std::vector<float> data;

    /** inCallData Matrix*/
    Eigen::MatrixXd inDataMat;
};

} // namespace megamol::infovis
