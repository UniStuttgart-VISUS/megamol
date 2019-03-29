/*
 * FileStreamProvider.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/FileStreamProvider.h"
#include "mmcore/param/FilePathParam.h"

#include "vislib/sys/Log.h"

#include <iostream>
#include <fstream>

namespace megamol {
namespace core {

    FileStreamProvider::FileStreamProvider() : filePath("file_path", "Output file path")
    {
        this->filePath << new core::param::FilePathParam("");
        this->MakeSlotAvailable(&this->filePath);
    }

    std::ostream& FileStreamProvider::GetStream()
    {
        if (!this->stream.is_open())
        {
            // Open file for writing
            this->stream.open(this->filePath.Param<core::param::FilePathParam>()->Value(), std::ios_base::out | std::ios_base::binary);

            if (!this->stream.good())
            {
                vislib::sys::Log::DefaultLog.WriteWarn("Unable to open file '%s' for writing!",
                    this->filePath.Param<core::param::FilePathParam>()->Value());
            }
        }

        return this->stream;
    }

}
}