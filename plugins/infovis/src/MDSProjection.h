#ifndef MEGAMOL_INFOVIS_MDSPROJECTION_H_INCLUDED
#define MEGAMOL_INFOVIS_MDSPROJECTION_H_INCLUDED

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/table/TableDataCall.h"


namespace megamol {
namespace infovis {

using namespace megamol::core;

class MDSProjection : public core::Module {
public:
    /** Return module class name */
    static inline const char* ClassName(void) { return "MDSProjection"; }

    /** Return module class description */
    static inline const char* Description(void) {
        return "Multidimensional scaling, i.e., a non-linear dimensionality reduction technique, by preserving "
               "distances/dissimilarities approximately";
    }

    /** Module is always available */
    static inline bool IsAvailable(void) { return true; }

    /** Constructor */
    MDSProjection(void);

    /** Destructor */
    virtual ~MDSProjection(void);

    static Eigen::MatrixXd euclideanDissimilarityMatrix(Eigen::MatrixXd dataMatrix);

    static Eigen::MatrixXd classicMds(Eigen::MatrixXd squaredDissimilarityMatrix, int outputDimension);

    static Eigen::MatrixXd smacofMds(Eigen::MatrixXd squaredDissimilarityMatrix, int outputDimension = 2,
        int countSteps = 100, Eigen::MatrixXd weightsMatrix = Eigen::MatrixXd::Ones(1, 1), double tolerance = 1e-3);

    static Eigen::MatrixXd ordinalMds(Eigen::MatrixXd squaredDissimilarityMatrix, int outputDimension = 2,
        int countSteps = 100, Eigen::MatrixXd weightsMatrix = Eigen::MatrixXd::Ones(1, 1), double tolerance = 1e-3);

    static double stress(Eigen::MatrixXd dissimilarityMatrix, Eigen::MatrixXd dataPointsMatrix,
        Eigen::MatrixXd weightsMatrix = Eigen::MatrixXd::Ones(1, 1));

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

private:
    static Eigen::MatrixXd bMatrix(Eigen::MatrixXd X, Eigen::MatrixXd W, Eigen::MatrixXd dissimilarityMatrix);

    static Eigen::MatrixXd vMatrix(Eigen::MatrixXd W);

    /** Data callback */
    bool getDataCallback(core::Call& c);

    bool getHashCallback(core::Call& c);

    bool dataProjection(megamol::stdplugin::datatools::table::TableDataCall* inCall);

    /** Data output slot */
    CalleeSlot dataOutSlot;

    /** Data output slot */
    CallerSlot dataInSlot;

    /** Parameter slot for target number of dimensions */
    ::megamol::core::param::ParamSlot reduceToNSlot;

    /** ID of the current frame */
    // int frameID; //TODO: unknown

    /** Hash of the current data */
    size_t datahash;
    size_t dataInHash;

    /** Vector storing information about columns */
    std::vector<megamol::stdplugin::datatools::table::TableDataCall::ColumnInfo> columnInfos;

    /** Vector stroing the actual float data */
    std::vector<float> data;
};

} // namespace infovis
} // namespace megamol


#endif
