/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::core {

/**
 * Call transporting a callback for writing data.
 *
 * @author Alexander Straub
 */
template<typename FunctionT>
class AbstractCallbackCall : public Call {

public:
    /**
     * Set the callback
     *
     * @param Callback New callback
     */
    void SetCallback(FunctionT callback) {
        this->callback = callback;
    }

    /**
     * Get the stored callback
     *
     * @return Callback
     */
    FunctionT GetCallback() const {
        return this->callback;
    }

protected:
    /**
     * Constructor
     */
    AbstractCallbackCall() {}

private:
    /** Store callback */
    FunctionT callback;
};

} // namespace megamol::core
