#include "stdafx.h"
#include "FSOctree.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "mmpld.h"

#include <sys/stat.h>
#include <filesystem>
#include <map>

megamol::stdplugin::datatools::FSOctreeMMPLD::FSOctreeMMPLD()
    : outBoxesSlot("outBoxes", "Output")
    , pathSlot("path", "where the files are located")
    , filenamePrefixSlot("filenamePrefix", "set the filename prefix")
    , extensionSlot("fileExtension", "set the filename extension")
    , maxSearchDepthSlot("maxDepth", "set how far to recurse to find the root node") {
    outBoxesSlot.SetCallback(megamol::geocalls::LinesDataCall::ClassName(),
        megamol::geocalls::LinesDataCall::FunctionName(0), &FSOctreeMMPLD::getDataCallback);
    outBoxesSlot.SetCallback(megamol::geocalls::LinesDataCall::ClassName(),
        megamol::geocalls::LinesDataCall::FunctionName(1), &FSOctreeMMPLD::getExtentCallback);
    MakeSlotAvailable(&outBoxesSlot);

    pathSlot << new core::param::StringParam("");
    MakeSlotAvailable(&pathSlot);

    filenamePrefixSlot << new core::param::StringParam("");
    MakeSlotAvailable(&filenamePrefixSlot);

    extensionSlot << new core::param::StringParam("");
    MakeSlotAvailable(&extensionSlot);

    maxSearchDepthSlot << new core::param::IntParam(10, 1, 50);
    MakeSlotAvailable(&maxSearchDepthSlot);
}


megamol::stdplugin::datatools::FSOctreeMMPLD::~FSOctreeMMPLD() { this->Release(); }


bool megamol::stdplugin::datatools::FSOctreeMMPLD::create() { return true; }


void megamol::stdplugin::datatools::FSOctreeMMPLD::release() {}


std::vector<uint8_t> octants_from_filename(const std::filesystem::path& file, const std::string& ext, const std::string& prefix) {
    std::vector<uint8_t> ret;
    auto n = file.stem();
    auto e = file.extension();
    if (e == ext) {
        auto pos = n.string().find_first_of(prefix);
        if (pos != std::string::npos) {
            auto nstr = n.string();
            auto treeStuff = nstr.substr(prefix.length(), nstr.length() - prefix.length());
            int d = std::count(treeStuff.begin(), treeStuff.end(), '_');
            ret.reserve(d);
            pos = treeStuff.find('_');
            while (pos != std::string::npos) {
                auto bits = treeStuff.substr(pos + 1, 3);
                int i = std::stoi(bits);
                int val = (i / 100) * 4 + (i % 100) / 10 * 2 + (i % 10);
                ret.push_back(val);
                pos = treeStuff.find('_', pos + 1);
            }
        }
    }
    return ret;
}

bool megamol::stdplugin::datatools::FSOctreeMMPLD::assertData(megamol::geocalls::LinesDataCall& outCall) {
    if (filenamePrefixSlot.IsDirty() || extensionSlot.IsDirty()) {

        std::string path = pathSlot.Param<core::param::StringParam>()->Value().PeekBuffer();
        std::string prefix = filenamePrefixSlot.Param<core::param::StringParam>()->Value().PeekBuffer();
        std::string ext = extensionSlot.Param<core::param::StringParam>()->Value().PeekBuffer();
        int maxDepth = maxSearchDepthSlot.Param<core::param::IntParam>()->Value();

        struct stat statbuf;

        //int minDepth = 
        for (auto& file: std::filesystem::directory_iterator(path)) {
            auto oct = octants_from_filename(file.path(), ext, prefix);
            if (oct.size() > 0) {
                octree_node::insert_node(this->tree, oct, file.path().stem().string());
            }
        }


        //bool anythingFound = false;
        //int minDepth = 0;
        //std::string currPrefix = prefix;
        //int currdepth = -1;
        //while(true) {
        //    for (int x = 0; x < 8; x++) {
        //        std::string filename = prefix + "_";
        //        filename += ((x & 4) > 0) ? "1" : "0";
        //        filename += ((x & 2) > 0) ? "1" : "0";
        //        filename += ((x & 1) > 0) ? "1" : "0";
        //        filename += ext;

        //        vislib::sys::Log::DefaultLog.WriteInfo("constructed filename: %s", filename.c_str());
        //        if (stat(filename.c_str(), &statbuf) == 0) {
        //            minDepth = currdepth;
        //            break;
        //        }
        //    }
        //    if (minDepth != -1 || ++currdepth > maxDepth) break;
        //    prefix += "_000";
        //}
        //vislib::sys::Log::DefaultLog.WriteInfo("octree root: %s", rootName.c_str());

        filenamePrefixSlot.ResetDirty();
        extensionSlot.ResetDirty();
    }
    
    return true;
}


bool megamol::stdplugin::datatools::FSOctreeMMPLD::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    return assertData(*outCall);
}


bool megamol::stdplugin::datatools::FSOctreeMMPLD::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    return assertData(*outCall);
}
