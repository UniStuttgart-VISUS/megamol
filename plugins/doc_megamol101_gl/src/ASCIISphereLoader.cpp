/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#include "ASCIISphereLoader.h"

#include <fstream>
#include <string>

#include "CallSpheres.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"

using namespace megamol;
using namespace megamol::megamol101_gl;

/*
 * ASCIISphereLoader::ASCIISphereLoader
 */
ASCIISphereLoader::ASCIISphereLoader()
        : core::Module()
        , filenameSlot("filename", "The path to the file that contains the data to be loaded")
        , getDataSlot("getdata", "The slot publishing the loaded data") {
    // TUTORIAL: A name and a description for each slot (CallerSlot, CalleeSlot, ParamSlot) has to be given in the
    // constructor initializer list

    // TUTORIAL: For each CalleeSlot all callback functions have to be set
    this->getDataSlot.SetCallback(CallSpheres::ClassName(), "GetData", &ASCIISphereLoader::getDataCallback);
    this->getDataSlot.SetCallback(CallSpheres::ClassName(), "GetExtent", &ASCIISphereLoader::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    // TUTORIAL: For each ParamSlot a default value has to be set
    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    // TUTORIAL: Each slot that shall be visible in the GUI has to be made available by this->MakeSlotAvailable(...)

    this->datahash = 0;
    this->numSpheres = 0;

    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
}

/*
 * ASCIISphereLoader::ASCIISphereLoader
 */
ASCIISphereLoader::~ASCIISphereLoader() {
    this->Release();
}

/*
 * ASCIISphereLoader::assertData
 */
void ASCIISphereLoader::assertData() {
    // we only want to reload the data if the filename has changed
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->spheres.clear();
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

        bool retval = false;
        try {
            // load the data from file
            retval =
                this->load(this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        } catch (vislib::Exception ex) {
            // a known vislib exception was raised
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unexpected exception: %s at (%s, %d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
            retval = false;
        } catch (...) {
            // an unknown exception was raised
            megamol::core::utility::log::Log::DefaultLog.WriteError("Unexpected exception: unkown exception\n");
            retval = false;
        }

        if (retval) {
            // standard case. The file has been successfully loaded.
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("Loaded %I64u spheres from file \"%s\"", numSpheres,
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Failed to load file \"%s\"",
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
            // we are in an erronous state, clean up everything
            this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
            this->numSpheres = 0;
            this->spheres.clear();
        }
        // the data has changed, so change the data hash, too
        this->datahash++;
    }
}

/*
 * ASCIISphereLoader::~ASCIISphereLoader
 */
bool ASCIISphereLoader::create() {
    // intentionally empty
    return true;
}

/*
 * ASCIISphereLoader::getDataCallback
 */
bool ASCIISphereLoader::getDataCallback(core::Call& caller) {
    CallSpheres* cs = dynamic_cast<CallSpheres*>(&caller);
    if (cs == nullptr)
        return false;

    this->assertData();

    cs->SetDataHash(this->datahash);
    cs->SetData(this->numSpheres, this->spheres.data());
    return true;
}

/*
 * ASCIISphereLoader::getExtentCallback
 */
bool ASCIISphereLoader::getExtentCallback(core::Call& caller) {
    CallSpheres* cs = dynamic_cast<CallSpheres*>(&caller);
    if (cs == nullptr)
        return false;

    this->assertData();

    cs->SetDataHash(this->datahash);
    cs->SetExtent(1, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(), this->bbox.Top(),
        this->bbox.Front());
    return true;
}

/*
 * ASCIISphereLoader::load
 */
bool ASCIISphereLoader::load(const vislib::TString& filename) {

    if (filename.IsEmpty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("No file to load (filename empty)");
        return true;
    }

    this->bbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    std::string line;
    std::ifstream file(T2A(filename));
    if (file.is_open()) {
        // the file is open, make the data storage ready to be written to
        this->spheres.clear();
        this->numSpheres = 0;
        int64_t lineNum = 0;

        // read it line by line
        while (std::getline(file, line)) {
            lineNum++;
            vislib::StringA lineA(line.c_str());
            if (lineA.IsEmpty())
                continue;
            lineA.TrimSpaces();
            if (lineA.StartsWith("#"))
                continue;                                                    // this is a comment, move on
            auto result = vislib::StringTokeniserA::Split(lineA, ",", true); // split the line string by commas
            if (result.Count() < 4) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "Line %u of '%s' is malformed", static_cast<unsigned int>(lineNum), T2A(filename).PeekBuffer());
                continue;
            }
            float values[4];
            bool error = false;
            for (int i = 0; i < 4; i++) { // read all 4 values
                auto resString = result[i];

                if (resString.Contains("#")) { // check if the line contains a comment after the values
                    if (i != 3) {
                        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Line %u of '%s' is malformed",
                            static_cast<unsigned int>(lineNum), T2A(filename).PeekBuffer());
                        error = true;
                    }
                    resString = resString.Substring(0, resString.Find("#") - 1);
                }

                try {
                    values[i] = std::stof(std::string(resString));
                } catch (...) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "The value with index %i in line %u of '%s' is malformed", i,
                        static_cast<unsigned int>(lineNum), T2A(filename).PeekBuffer());
                    error = true;
                }
            }
            if (error) { // if there was no error, we have a new sphere
                continue;
            } else {
                for (int i = 0; i < 4; i++) {
                    this->spheres.push_back(values[i]);
                }

                // update the bounding box, so that it contains the new sphere
                vislib::math::Cuboid<float> sphereBBox(values[0] - values[3], values[1] - values[3],
                    values[2] - values[3], values[0] + values[3], values[1] + values[3], values[2] + values[3]);
                if (numSpheres == 0) {
                    this->bbox = sphereBBox;
                } else {
                    this->bbox.Union(sphereBBox);
                }

                this->numSpheres++;
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to open file");
        return false;
    }

    return true;
}

/*
 * ASCIISphereLoader::release
 */
void ASCIISphereLoader::release() {
    this->spheres.clear();
}
