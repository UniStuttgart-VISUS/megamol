/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <string>
#include <utility>

#include "mmcore/AbstractSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::core {

/**
 * Abstract class for providing parameters for writers.
 *
 * @author Alexander Straub
 */
class AbstractWriterParams {

protected:
    /**
     * Constructor
     *
     * @param makeSlotAvailable Function to make the slots available in the
     *                          module partially derived from this class
     */
    AbstractWriterParams(std::function<void(AbstractSlot* slot)> makeSlotAvailable);

    /**
     * Get the next filename, based on the set parameters.
     *
     * @return The generated filename and the success in setting it
     */
    std::pair<bool, std::string> getNextFilename();

private:
    /**
     * Handle changed mode.
     *
     * @return Success
     */
    bool modeChanged(param::ParamSlot&);

    /** File path parameter */
    param::ParamSlot filePathSlot;

    /** Choice on how to name the file */
    param::ParamSlot writeMode;

    /** Start counter for postfix */
    param::ParamSlot countStart;

    /** Length of the counter */
    param::ParamSlot countLength;

    /** The counter */
    unsigned int count;
};

} // namespace megamol::core
