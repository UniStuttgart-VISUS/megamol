#ifndef MEGAMOL_UMAP_MODULE_H_INCLUDED
#define MEGAMOL_UMAP_MODULE_H_INCLUDED

#include "datatools/table/TableDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace infovis {

using namespace megamol::core;

class UMAProjection : public core::Module {
public:
    /** Return module class name */
    static inline const char* ClassName(void) {
        return "UMAProjection";
    }

    /** Return module class description */
    static inline const char* Description(void) {
        return "Uniform Manifold Approximation and Projection (UMAP), i.e., "
               "a fairly flexible non-linear dimension reduction algorithm";
    }

    /** Module is always available */
    static inline bool IsAvailable(void) {
        return true;
    }

    /** Constructor */
    UMAProjection(void);

    /** Destructor */
    virtual ~UMAProjection(void);

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

private:
    /** Data callback */
    bool getDataCallback(core::Call& c);

    bool getHashCallback(core::Call& c);

    bool project(megamol::datatools::table::TableDataCall* inCall);

    /** Data output slot */
    CalleeSlot dataOutSlot;

    /** Data output slot */
    CallerSlot dataInSlot;

    /** Parameter slot for column selection */
    ::megamol::core::param::ParamSlot nDimsSlot;
    ::megamol::core::param::ParamSlot randomSeedSlot;
    ::megamol::core::param::ParamSlot nEpochsSlot;
    ::megamol::core::param::ParamSlot learningRateSlot;
    ::megamol::core::param::ParamSlot localConnectivitySlot;
    ::megamol::core::param::ParamSlot bandwidthSlot;
    ::megamol::core::param::ParamSlot mixRatioSlot;
    ::megamol::core::param::ParamSlot spreadSlot;
    ::megamol::core::param::ParamSlot minDistSlot;
    ::megamol::core::param::ParamSlot aSlot;
    ::megamol::core::param::ParamSlot bSlot;
    ::megamol::core::param::ParamSlot repulsionStrengthSlot;
    ::megamol::core::param::ParamSlot initializeSlot;
    ::megamol::core::param::ParamSlot negativeSampleRateSlot;
    ::megamol::core::param::ParamSlot nNeighborsSlot;

    /** ID of the current frame */
    // int frameID; //TODO: unknown

    /** Hash of the current data */
    size_t datahash;
    size_t dataInHash;

    /** Vector storing information about columns */
    std::vector<megamol::datatools::table::TableDataCall::ColumnInfo> columnInfos;

    /** Vector stroing the actual float data */
    std::vector<float> data;
};

} // namespace infovis
} // namespace megamol


#endif
