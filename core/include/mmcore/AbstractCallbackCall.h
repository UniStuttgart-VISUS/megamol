/*
 * AbstractCallbackCall.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {

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

} // namespace core
} // namespace megamol
