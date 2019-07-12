/*
 * FileStreamProvider.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractStreamProvider.h"
#include "mmcore/param/ParamSlot.h"

#include <iostream>
#include <fstream>

namespace megamol {
namespace core {

    /**
    * Provides a stream.
    *
    * @author Alexander Straub
    */
    class MEGAMOLCORE_API FileStreamProvider : public AbstractStreamProvider
    {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName() { return "FileStreamProvider"; }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description() { return "Provides a file stream"; }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable() { return true; }

        /**
         * Disallow usage in quickstarts.
         *
         * @return false
         */
        static bool SupportQuickstart() { return false; }

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
        virtual std::iostream& GetStream() override;

    private:
        /** File path parameter */
        core::param::ParamSlot filePath;

        /** File path parameter */
        core::param::ParamSlot append;

        /** File stream */
        std::fstream stream;
    };

}
}