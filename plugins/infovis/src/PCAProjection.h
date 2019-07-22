#ifndef MEGAMOL_PRINCIPAL_COMPONENT_ANALYSIS_H_INCLUDED
#define MEGAMOL_PRINCIPAL_COMPONENT_ANALYSIS_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/table/TableDataCall.h"


namespace megamol {
namespace infovis {

using namespace megamol::core;

class PCAProjection : public core::Module {
public:
    /** Return module class name */
    static inline const char* ClassName(void) { return "PCAProjection"; }

    /** Return module class description */
    static inline const char* Description(void) {
        return "Principal component analysis, i.e., a linear and orthogonal dimensionality reduction technique";
    }

    /** Module is always available */
    static inline bool IsAvailable(void) { return true; }

    /** Constructor */
    PCAProjection(void);

    /** Destructor */
    virtual ~PCAProjection(void);

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

private:
    /** Data callback */
    bool getDataCallback(core::Call& c);

    bool getHashCallback(core::Call& c);

    bool project(megamol::stdplugin::datatools::table::TableDataCall* inCall);

    /** Data output slot */
    CalleeSlot dataOutSlot;

    /** Data output slot */
    CallerSlot dataInSlot;

    /** Parameter slot for column selection */
    ::megamol::core::param::ParamSlot reduceToNSlot;
    ::megamol::core::param::ParamSlot scaleSlot;
    ::megamol::core::param::ParamSlot centerSlot;

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
