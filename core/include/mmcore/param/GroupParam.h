/*
 * GroupParam.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/api/MegaMolCore.std.h"
#include "StringParam.h"
#include "vislib/String.h"
#include "vislib/tchar.h"

namespace megamol {
namespace core {
namespace param {

    /**
     * class for string parameter objects
     */
    class MEGAMOLCORE_API GroupParam : public StringParam {
    public:

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param visible If 'true' the parameter is visible in the gui.
         */
        GroupParam(const vislib::StringA& initVal) : StringParam(initVal) {}

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param visible If 'true' the parameter is visible in the gui.
         */
        GroupParam(const vislib::StringW& initVal) : StringParam(initVal) {}

        /**
         * Ctor.
         *
         * @param initVal The initial value
         */
        GroupParam(const char *initVal) : StringParam(initVal) {}

        /**
         * Ctor.
         *
         * @param initVal The initial value
         */
        GroupParam(const wchar_t *initVal) : StringParam(initVal) {}

        /**
         * Returns a machine-readable definition of the parameter.
         *
         * @param outDef A memory block to receive a machine-readable
         *               definition of the parameter.
         */
        virtual void Definition(vislib::RawStorage& outDef) const {
            outDef.AssertSize(7);
            memcpy(outDef.AsAt<char>(0), "MMSGRP", 7);
        }

    };

    /**
     * class for string parameter objects
     */
    class MEGAMOLCORE_API GroupEndParam : public StringParam {
    public:

        /**
         * Ctor.
         */
        GroupEndParam() : StringParam("") {}

        /**
         * Returns a machine-readable definition of the parameter.
         *
         * @param outDef A memory block to receive a machine-readable
         *               definition of the parameter.
         */
        virtual void Definition(vislib::RawStorage& outDef) const {
            outDef.AssertSize(8);
            memcpy(outDef.AsAt<char>(0), "MMEGRP", 8);
        }
    };

}
}
}
