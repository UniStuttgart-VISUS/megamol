#ifndef MEGAMOL_DATATOOLS_LINESTOCSV_H_INCLUDED
#define MEGAMOL_DATATOOLS_LINESTOCSV_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class LinesToFloatTable : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static char const* ClassName(void) { return "LinesToFloatTable"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static char const* Description(void) { return "Module converting lines into a floattable"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    LinesToFloatTable(void);

    /** Dtor. */
    virtual ~LinesToFloatTable(void);

protected:
    /** Lazy initialization */
    virtual bool create(void);

    /** Deferred destruction */
    virtual void release(void);

private:
    bool getDataCallback(megamol::core::Call& c);

    bool assertData(megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall);

    megamol::core::CalleeSlot dataOutSlot;

    megamol::core::CallerSlot dataInSlot;

    std::vector<megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnInfo> columnInfos;

    unsigned int frameID;

    size_t datahash;

    std::vector<float> data;

}; /* end class LinesToFloatTable */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_LINESTOCSV_H_INCLUDED */
