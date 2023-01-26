#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"

namespace megamol::datatools {

class CSVWriter : public core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CSVWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "CSV file writer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    CSVWriter();

    ~CSVWriter() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

    /**
     * The main function
     *
     * @return True on success
     */
    bool run(void) override;

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    bool getCapabilities(core::DataWriterCtrlCall& call) override;

private:
    core::CallerSlot _data_in_slot;

    core::param::ParamSlot _filename_slot;
};

} // end namespace megamol::datatools
