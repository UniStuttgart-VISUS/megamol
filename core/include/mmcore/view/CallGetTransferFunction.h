/*
 * AbstractCallGetTransferFunction.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */


#pragma once

#include "AbstractCallGetTransferFunction.h"

namespace megamol {
namespace core {
namespace view {


/**
 * Call for accessing a transfer function.
 */
class CallGetTransferFunction : public AbstractCallGetTransferFunction {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallGetTransferFunction";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call for a 1D transfer function";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetTexture";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallGetTransferFunction(void) : AbstractCallGetTransferFunction() {}

    /** Dtor. */
    virtual ~CallGetTransferFunction(void) {}
};


/** Description class typedef */
typedef core::factories::CallAutoDescription<CallGetTransferFunction> CallGetTransferFunctionDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
