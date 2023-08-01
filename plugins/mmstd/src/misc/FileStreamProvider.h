/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmstd/generic/AbstractStreamProvider.h"

#include <fstream>
#include <iostream>

namespace megamol::core {

/**
 * Provides a stream.
 *
 * @author Alexander Straub
 */
class FileStreamProvider : public AbstractStreamProvider {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "FileStreamProvider";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Provides a file stream";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Constructor
     */
    FileStreamProvider();

protected:
    /**
     * Callback function providing the stream.
     *
     * @return Stream
     */
    std::iostream& GetStream() override;

private:
    /** File path parameter */
    core::param::ParamSlot filePath;

    /** File path parameter */
    core::param::ParamSlot append;

    /** File stream */
    std::fstream stream;
};

} // namespace megamol::core
