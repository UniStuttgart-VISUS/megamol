/*
 * CPERAWDataSource.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <array>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <sys/stat.h>

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {
namespace io {

/** Module to read cpe point dumps as introduced with cpelib 4513dfd9fd8efa9282848bbe2ab4f53053d753b8. */
class CPERAWDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "CPERAWDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Data reader module for cpelib dumps.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    CPERAWDataSource(void);

    /** dtor */
    virtual ~CPERAWDataSource(void);

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

private:

    /**
     * Reads and decompressed PBS file
     *
     * @param filename Path to the PBS file without extension
     * @param data Buffer to store the uncompressed file content
     *
     * @return True, if file was successfully read
     */
    bool assertData();

    bool isDirty(void);

    void resetDirty(void);

    /**
     * Callback receiving the update of the radius parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool radiusChanged(core::param::ParamSlot& slot);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& c);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& c);

    /** The file name */
    core::param::ParamSlot filenameSlot;

    /** the point size */
    core::param::ParamSlot radiusSlot;

    /** The slot for requesting data */
    core::CalleeSlot getData;

    /** Buffer holding the data of the PBS file */
    std::vector<char> data;

    /** The data set data hash */
    unsigned int dataHash = 0;

    bool newFile = false;

    size_t numPoints;

    size_t const headerLen = 48;
    size_t const pointStride = 27;

    std::array<float, 6> localBBox, globalBBox, globalCBox;
};

} // namespace io
} // namespace datatools
} // namespace stdplugin
} // namespace megamol
