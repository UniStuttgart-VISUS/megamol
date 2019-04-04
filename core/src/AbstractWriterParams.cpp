/*
 * AbstractWriterParams.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/AbstractSlot.h"
#include "mmcore/AbstractWriterParams.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/sys/Log.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace megamol {
namespace core {

    AbstractWriterParams::AbstractWriterParams(std::function<void(AbstractSlot *slot)> makeSlotAvailable) :
        filePathSlot("outputFile", "Path to the file which should be written"),
        writeMode("outputMode", "Choice on how to write and name the output file"),
        countStart("outputCountStart", "Start of the counter (modifying this value resets the counter!)"),
        countLength("outputCountLength", "Length of the counter (modyfying this value resets the counter!)"),
        count(0) {

        this->filePathSlot << new param::FilePathParam("");
        makeSlotAvailable(&this->filePathSlot);

        this->writeMode << new param::EnumParam(0);
        this->writeMode.Param<param::EnumParam>()->SetTypePair(0, "Overwrite existing file");
        this->writeMode.Param<param::EnumParam>()->SetTypePair(1, "Abort if file exists");
        this->writeMode.Param<param::EnumParam>()->SetTypePair(2, "Add count postfix");
        this->writeMode.Param<param::EnumParam>()->SetTypePair(3, "Add timestamp postfix");
        this->writeMode.SetUpdateCallback(this, &AbstractWriterParams::modeChanged);
        makeSlotAvailable(&this->writeMode);

        this->countStart << new param::IntParam(0);
        makeSlotAvailable(&this->countStart);

        this->countLength << new param::IntParam(5);
        makeSlotAvailable(&this->countLength);

        modeChanged(this->writeMode);
    }

    std::pair<bool, std::string> AbstractWriterParams::getNextFilename() {
        const std::string filepath_param = this->filePathSlot.template Param<param::FilePathParam>()->Value().PeekBuffer();

        const std::string filepath = filepath_param.find_last_of("/\\") != std::string::npos ? filepath_param.substr(0, filepath_param.find_last_of("/\\") + 1) : "";
        const std::string filename = filepath_param.find_last_of("/\\") != std::string::npos ? filepath_param.substr(filepath_param.find_last_of("/\\") + 1) : filepath_param;
        const std::string filename_we = filename.find_last_of('.') != std::string::npos ? filename.substr(0, filename.find_last_of('.')) : filename;
        const std::string file_ext = filename.find_last_of('.') != std::string::npos ? filename.substr(filename.find_last_of('.')) : "";

        std::string postfix = "";

        if (this->writeMode.Param<param::EnumParam>()->Value() == 1 && std::ifstream(filepath_param).good()) {
            // Do not overwrite file
            vislib::sys::Log::DefaultLog.WriteWarn("Did not write file '%s'. File already exists.", filepath_param.c_str());

            return std::make_pair(false, std::string(""));
        }
        else if (this->writeMode.Param<param::EnumParam>()->Value() == 2) {
            // Set counter postfix
            if (this->filePathSlot.IsDirty() || this->countStart.IsDirty() || this->countLength.IsDirty()) {
                this->count = static_cast<unsigned int>(this->countStart.Param<param::IntParam>()->Value());

                this->filePathSlot.ResetDirty();
                this->countStart.ResetDirty();
                this->countLength.ResetDirty();
            }

            std::string zeroes = "";
            for (int i = 0; i < this->countLength.Param<param::IntParam>()->Value(); ++i) {
                zeroes += "0";
            }

            const std::string count_string = zeroes + std::to_string(this->count);

            postfix = "_" + count_string.substr(count_string.length() - this->countLength.Param<param::IntParam>()->Value());

            ++this->count;
        }
        else if (this->writeMode.Param<param::EnumParam>()->Value() == 3) {
            // Set timestamp postfix
            std::time_t now = std::time(nullptr);

            std::stringstream ss;
            ss << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");

            std::chrono::high_resolution_clock hrc;
            auto now_hrc = hrc.now().time_since_epoch();

            std::string id = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now_hrc).count() % 1000);

            postfix = "_" + ss.str() + "_" + id;
        }

        return std::make_pair(true, filepath + filename_we + postfix + file_ext);
    }

    bool AbstractWriterParams::modeChanged(param::ParamSlot&) {
        this->countStart.Parameter()->SetGUIVisible(this->writeMode.Param<param::EnumParam>()->Value() == 2);
        this->countLength.Parameter()->SetGUIVisible(this->writeMode.Param<param::EnumParam>()->Value() == 2);

        this->count = static_cast<unsigned int>(this->countStart.Param<param::IntParam>()->Value());

        return true;
    }

}
}