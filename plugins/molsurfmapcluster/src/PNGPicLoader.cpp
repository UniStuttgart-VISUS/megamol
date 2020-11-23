/*
 * PNGPicLoader.cpp
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PNGPicLoader.h"
#include <fstream>
#include <string>
#include "CallPNGPics.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringTokeniser.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * PNGPicLoader::PNGPicLoader
 */
PNGPicLoader::PNGPicLoader(void)
        : core::Module()
        , filenameSlot("filename", "The path to the file that contains the PNG-Filepaths to be loaded")
        , getDataSlot("getdata", "The slot publishing the loaded data") {

    // For each CalleeSlot all callback functions have to be set
    this->getDataSlot.SetCallback(CallPNGPics::ClassName(), "GetData", &PNGPicLoader::getDataCallback);
    this->getDataSlot.SetCallback(CallPNGPics::ClassName(), "GetExtent", &PNGPicLoader::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    // For each ParamSlot a default value has to be set
    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->datahash = 0;
    this->numPics = 0;

    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
}

/*
 * PNGPicLoader::PNGPicLoader
 */
PNGPicLoader::~PNGPicLoader(void) {
    this->Release();
}

/*
 * PNGPicLoader::assertData
 */
void PNGPicLoader::assertData(void) {
    // we only want to reload the data if the filename has changed
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->pngpics.clear();
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

        bool loaded = false;
        try {
            // Load PNG-Pictures contained in File
            loaded = this->load(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
        } catch (vislib::Exception ex) {
            // a known vislib exception was raised
            core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_ERROR,
                "Unexpected exception: %s at (%s, %d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
            loaded = false;
        } catch (...) {
            // an unknown exception was raised
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_ERROR, "Unexpected exception: unkown exception\n");
            loaded = false;
        }

        if (loaded) {
            // All PNG-Pics has been successfully loaded
            core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO,
                "Loaded %I64u PNG-Pictures from file \"%s\"", numPics,
                vislib::StringA(this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        } else {
            // Picture not successfully loaded
            core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_ERROR,
                "Failed to load file \"%s\"",
                vislib::StringA(this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
            // we are in an erronous state, clean up everything
            this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
            this->numPics = 0;
            this->pngpics.clear();
        }

        // the data has changed, so change the data hash, too
        this->datahash++;
    }
}

/*
 * PNGPicLoader::~PNGPicLoader
 */
bool PNGPicLoader::create(void) {
    // intentionally empty
    return true;
}

/*
 * PNGPicLoader::getDataCallback
 */
bool PNGPicLoader::getDataCallback(core::Call& caller) {

    CallPNGPics* cs = dynamic_cast<CallPNGPics*>(&caller);
    if (cs == nullptr)
        return false;

    this->assertData();

    cs->SetDataHash(this->datahash);
    cs->SetData(this->numPics, this->pngpics.data());
    return true;
}

/*
 * PNGPicLoader::getExtentCallback
 */
bool PNGPicLoader::getExtentCallback(core::Call& caller) {

    CallPNGPics* cs = dynamic_cast<CallPNGPics*>(&caller);
    if (cs == nullptr)
        return false;

    this->assertData();

    cs->SetDataHash(this->datahash);
    cs->SetExtent(1, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(), this->bbox.Top(),
        this->bbox.Front());
    return true;
}

/*
 * PNGPicLoader::load
 */
bool PNGPicLoader::load(const vislib::TString& filename) {

    if (filename.IsEmpty()) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_INFO, "No file to load (filename empty)");
        return true;
    }

    this->bbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    std::string line;
    std::ifstream file(filename.PeekBuffer());
    if (file.is_open()) {
        // Make all ready and clean old up
        this->pngpics.clear();
        this->numPics = 0;
        int64_t lineNum = 0;

        // read file
        while (std::getline(file, line)) {
            lineNum++;
            vislib::StringA lineA(line.c_str());
            if (lineA.IsEmpty())
                continue; // Empty line move on
            lineA.TrimSpaces();
            if (lineA.StartsWith("#"))
                continue; // Comment move on

            // try to load PNG File from String
            core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO, "Load Picture %s", lineA);

            // OPen File and check for PNG-Picture
            FILE* fp = fopen(lineA, "rb");

            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
            if (!png)
                abort();

            png_infop info = png_create_info_struct(png);
            if (!info)
                abort();

            if (setjmp(png_jmpbuf(png)))
                abort();

            // Init PNG-Pic
            png_set_palette_to_rgb(png);
            png_init_io(png, fp);
            png_read_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

            // Create PNG-Pic with infos
            PNGPIC* pic = new PNGPIC;

            // Set Filepath of pic
            pic->name = lineA;

            // Set PNG Infos
            pic->fp = fp;
            pic->png = png;
            pic->info = info;
            pic->rows = png_get_rows(png, info);
            pic->width = png_get_image_width(png, info);
            pic->height = png_get_image_height(png, info);
            pic->render = false;
            pic->popup = false;
            pic->texture = nullptr;

            // Add picture to list of Pictures
            pngpics.push_back(*pic);
        }
        this->numPics = pngpics.size();
    }
    return true;
}

/*
 * PNGPicLoader::release
 */
void PNGPicLoader::release(void) {
    this->pngpics.clear();
}
