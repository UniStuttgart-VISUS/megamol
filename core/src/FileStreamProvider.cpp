/*
 * FileStreamProvider.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/FileStreamProvider.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"

#include "mmcore/utility/log/Log.h"

#include <fstream>
#include <iostream>

namespace megamol {
namespace core {

FileStreamProvider::FileStreamProvider()
        : filePath("file_path", "Output file path")
        , append("append", "Append instead of overwriting?") {
    this->filePath << new core::param::FilePathParam("", core::param::FilePathParam::Flag_File_ToBeCreated);
    this->MakeSlotAvailable(&this->filePath);

    this->append << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->append);
}

std::iostream& FileStreamProvider::GetStream() {
    if (!this->stream.is_open()) {
        // Open file for writing
        if (this->append.Param<core::param::BoolParam>()->Value()) {
            this->stream.open(this->filePath.Param<core::param::FilePathParam>()->Value(),
                std::ios_base::app | std::ios_base::binary);
        } else {
            this->stream.open(this->filePath.Param<core::param::FilePathParam>()->Value(),
                std::ios_base::out | std::ios_base::binary);
        }

        if (!this->stream.good()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("Unable to open file '%s' for writing!",
                this->filePath.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        }
    }

    return this->stream;
}

} // namespace core
} // namespace megamol
