/*
 * PNGPicLoader.cpp
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusteringLoader.h"
#include <filesystem>
#include <fstream>
#include <string>
#include "CallClusteringLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/StringTokeniser.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * PNGPicLoader::PNGPicLoader
 */
ClusteringLoader::ClusteringLoader(void)
        : core::Module()
        , filenameSlot("filename", "The path to the dot file")
        , getDataSlot("getdata", "The slot publishing the loaded data") {

    // For each CalleeSlot all callback functions have to be set
    this->getDataSlot.SetCallback(CallClusteringLoader::ClassName(), "GetData", &ClusteringLoader::getDataCallback);
    this->getDataSlot.SetCallback(CallClusteringLoader::ClassName(), "GetExtent", &ClusteringLoader::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    // For each ParamSlot a default value has to be set
    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->datahash = 0;

    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
}

/*
 * PNGPicLoader::PNGPicLoader
 */
ClusteringLoader::~ClusteringLoader(void) {
    this->Release();
}

/*
 * PNGPicLoader::assertData
 */
void ClusteringLoader::assertData(void) {
    // we only want to reload the data if the filename has changed
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
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
                "Loaded %I64u Pictrues from file \"%s\"", leaves.size(),
                vislib::StringA(this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        } else {
            // Picture not successfully loaded
            core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_ERROR,
                "Failed to load file \"%s\"",
                vislib::StringA(this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
            // we are in an erronous state, clean up everything
            this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }

        // the data has changed, so change the data hash, too
        this->datahash++;
    }
}

/*
 * ClusteringLoader::create
 */
bool ClusteringLoader::create(void) {
    // intentionally empty
    return true;
}

/*
 * ClusteringLoader::getDataCallback
 */
bool ClusteringLoader::getDataCallback(core::Call& caller) {

    CallClusteringLoader* cs = dynamic_cast<CallClusteringLoader*>(&caller);
    if (cs == nullptr)
        return false;

    this->assertData();

    cs->SetDataHash(this->datahash);
    cs->SetData(this->leaves.size(), this->leaves.data());
    return true;
}

/*
 * PNGPicLoader::getExtentCallback
 */
bool ClusteringLoader::getExtentCallback(core::Call& caller) {

    CallClusteringLoader* cs = dynamic_cast<CallClusteringLoader*>(&caller);
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
bool ClusteringLoader::load(const vislib::TString& filename) {

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
            if (lineA.StartsWith("graph"))
                continue; // Start line

            if (lineA.Contains("--")) {
                // Line is node

            } else if (lineA.Contains("rank")) {
                // Line is ranking => do nothing
            } else {
                // Line is leave
                HierarchicalClustering::CLUSTERNODE* node = new HierarchicalClustering::CLUSTERNODE();
                while (!lineA.IsEmpty()) {
                    if (lineA.Contains("[")) {
                        int pos = lineA.Find("[");
                        node->id = std::stoi(std::string(lineA.Substring(0, pos)));
                        node->pic = new PictureData();
                        lineA = lineA.Substring(pos + 1);
                    } else if (!lineA.StartsWith("]") && lineA.Contains(",")) {
                        int poskomma = lineA.Find(",");
                        vislib::StringA tmp = lineA.Substring(0, poskomma);
                        int pos = tmp.Find("=");
                        vislib::StringA tmpidentifier = tmp.Substring(0, pos);
                        tmpidentifier.TrimSpaces();

                        if (tmpidentifier.Equals("image")) {
                            auto s = tmp.Substring(pos + 1);
                            s.Trim("\"");
                            node->pic->path = s;
                            // Load pic...
                            core::utility::log::Log::DefaultLog.WriteMsg(
                                core::utility::log::Log::LEVEL_INFO, "Load Picture: %s", node->pic->path);
                            // OPen File and check for PNG-Picture
                            /*
                            FILE* fp = fopen(node->pic->name, "rb");

                            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
                            if (!png) abort();

                            png_infop info = png_create_info_struct(png);
                            if (!info) abort();

                            if (setjmp(png_jmpbuf(png))) abort();

                            // Init PNG-Pic
                            png_set_palette_to_rgb(png);
                            png_init_io(png, fp);
                            png_read_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);
                            */
                            auto fileSize = std::filesystem::file_size(node->pic->path);
                            std::vector<uint8_t> loadedFile;
                            loadedFile.resize(fileSize);
                            std::ifstream fp(node->pic->path, std::ios::binary);
                            if (!fp.is_open())
                                abort();
                            file.read(reinterpret_cast<char*>(loadedFile.data()), fileSize);
                            node->pic->image = new vislib::graphics::BitmapImage(); // ugly shit, but is not removable
                                                                                    // without major refactoring
                            if (!vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(
                                    *node->pic->image, loadedFile.data(), fileSize))
                                abort();
                            node->pic->image->Convert(vislib::graphics::BitmapImage::TemplateByteRGB);


                            // Set PNG Infos
                            node->pic->pdbid = std::filesystem::path(node->pic->path).stem().string();
                            node->pic->width = node->pic->image->Width();
                            node->pic->height = node->pic->image->Height();
                            node->pic->render = false;
                            node->pic->popup = false;
                            node->pic->texture = nullptr;

                        } else if (tmpidentifier.Equals("features")) {
                            vislib::StringA tmpvalue = tmp.Substring(pos + 1);
                            tmpvalue.Trim("\"");

                            node->features = new std::vector<double>();

                            while (tmpvalue.Contains(";")) {
                                tmpvalue.Trim(";");
                                int posfeatures = tmpvalue.Find(";");
                                double feature = 0;
                                if (posfeatures == -1) {
                                    feature = std::stod(std::string(tmpvalue));
                                } else {
                                    feature = std::stod(std::string(tmpvalue.Substring(0, posfeatures)));
                                }

                                node->features->push_back(feature);
                                tmpvalue = tmpvalue.Substring(posfeatures);
                            }
                        }
                        lineA = lineA.Substring(poskomma + 1);
                    } else {


                        lineA = lineA.EMPTY;
                    }
                }
                this->leaves.push_back(*node);
            }
        }
    }
    return true;
}

/*
 * ClusteringLoader::release
 */
void ClusteringLoader::release(void) {}
