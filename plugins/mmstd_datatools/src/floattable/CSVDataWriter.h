#ifndef MEGAMOL_DATATOOLS_CSVDATAWRITER_INCLUDED_H
#define MEGAMOL_DATATOOLS_CSVDATAWRITER_INCLUDED_H

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {

class CSVDataWriter : public megamol::core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "CSVDataWriter"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Float table data file writer"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) { return false; }

    /** Ctor. */
    CSVDataWriter(void);

    /** Dtor. */
    virtual ~CSVDataWriter(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The main function
     *
     * @return True on success
     */
    virtual bool run(void);

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    virtual bool getCapabilities(megamol::core::DataWriterCtrlCall& call);

private:
    /** The file name of the file to be written */
    megamol::core::param::ParamSlot filenameSlot;

    /** The slot asking for data */
    megamol::core::CallerSlot dataSlot;
}; /* end class CSVDataSource */

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_CSVDATAWRITER_INCLUDED_H */
