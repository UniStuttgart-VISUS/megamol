/*
 * FileStreamProvider.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/FileStreamProvider.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"

#include "vislib/sys/Log.h"

#include <iostream>
#include <fstream>

namespace megamol {
namespace core {

    FileStreamProvider::FileStreamProvider() : filePath("file_path", "Output file path"), append("append", "Append instead of overwriting?")
    {
        this->filePath << new core::param::FilePathParam("");
        this->MakeSlotAvailable(&this->filePath);

        this->append << new core::param::BoolParam(true);
        this->MakeSlotAvailable(&this->append);
    }

    std::iostream& FileStreamProvider::GetStream()
    {
        if (!this->stream.is_open())
        {
            // Open file for writing
            if (this->append.Param<core::param::BoolParam>()->Value())
            {
                this->stream.open(this->filePath.Param<core::param::FilePathParam>()->Value(), std::ios_base::app | std::ios_base::binary);
            }
            else
            {
                this->stream.open(this->filePath.Param<core::param::FilePathParam>()->Value(), std::ios_base::out | std::ios_base::binary);
            }

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