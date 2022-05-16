/*
 * Clustering.cpp
 *
 * Copyright (C) 2019 by Tobias Baur
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "Clustering.h"

#include "CallCluster.h"
#include "CallClusteringLoader.h"
#include "CallPNGPics.h"
#include "DistanceMatrixLoader.h"
#include "image_calls/Image2DCall.h"

#include <fstream>
#include <iostream>
#include <string>

#include "mmcore/param//ButtonParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/StringTokeniser.h"

using namespace megamol;
using namespace megamol::molsurfmapcluster;

/*
 * Clustering::Clustering
 */
Clustering::Clustering(void)
        : core::Module()
        , inSlotImageLoader("inImages", "Input slot for image data")
        , inSlotImageLoader2("inImages2", "Second input slot for image data")
        , inSlotImageLoader3("inImages3", "Third input slot for image data")
        , inSlotCLUSTERINGLoader("inClustering", "Input Slot for Clustering Data")
        , outSlot("outClusteringSlot", "OUtput slot for the Clustering")
        , dumpdot("Dump Dot-File", "")
        , dumpdotpath("File-Path for Dot-File", "")
        , distanceFilePath("distFile", "File-Path for the distance override csv file")
        , selectionmode("Mode for selection of similar nodes", "")
        , linkagemodeparam("Linkage Mode", "")
        , distancemultiplier("Distance Multiplier", "")
        , momentsmethode("Moments Methode", "")
        , featuresSlot1Param("feat::featuresCall1", "Path to an additional feature vector file for input call 1")
        , featuresSlot2Param("feat::featuresCall2", "Path to an additional feature vector file for input call 2")
        , featuresSlot3Param("feat::featuresCall3", "Path to an additional feature vector file for input call 3") {

    // Callee-Slot
    this->outSlot.SetCallback(CallClustering::ClassName(), "GetData", &Clustering::getDataCallback);
    this->outSlot.SetCallback(CallClustering::ClassName(), "GetExtent", &Clustering::getExtentCallback);
    this->MakeSlotAvailable(&this->outSlot);

    // Caller-Slot
    this->inSlotImageLoader.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->inSlotImageLoader);

    this->inSlotImageLoader2.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->inSlotImageLoader2);

    this->inSlotImageLoader3.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->inSlotImageLoader3);

    this->inSlotCLUSTERINGLoader.SetCompatibleCall<CallClusteringLoaderDescription>();
    this->MakeSlotAvailable(&this->inSlotCLUSTERINGLoader);

    // ParamSlot
    this->dumpdot.SetParameter(new megamol::core::param::ButtonParam(megamol::core::view::Key::KEY_D));
    this->MakeSlotAvailable(&this->dumpdot);

    this->dumpdotpath.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->dumpdotpath);

    this->distanceFilePath.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->distanceFilePath);

    core::param::EnumParam* cluster_mode_param =
        new core::param::EnumParam(static_cast<int>(ClusteringMode::EUCLIDIAN));
    ClusteringMode cluster_mode = ClusteringMode::COSINUS;
    cluster_mode_param->SetTypePair(cluster_mode, "Similarity Cosinus-Koeffizient");
    cluster_mode = ClusteringMode::DICE;
    cluster_mode_param->SetTypePair(cluster_mode, "Similarity Dice-Koeffizient");
    cluster_mode = ClusteringMode::JACCARD;
    cluster_mode_param->SetTypePair(cluster_mode, "Similarity Jaccard-Koeffizient");
    cluster_mode = ClusteringMode::OVERLAP;
    cluster_mode_param->SetTypePair(cluster_mode, "Similarity Overlap-Koeffizient");
    cluster_mode = ClusteringMode::CITYBLOCK;
    cluster_mode_param->SetTypePair(cluster_mode, "Distance City-Block-Mannhatten");
    cluster_mode = ClusteringMode::EUCLIDIAN;
    cluster_mode_param->SetTypePair(cluster_mode, "Distance Euclidian");
    cluster_mode = ClusteringMode::L3;
    cluster_mode_param->SetTypePair(cluster_mode, "Distance L3");
    cluster_mode = ClusteringMode::GOWER;
    cluster_mode_param->SetTypePair(cluster_mode, "Distance Gower");
    this->selectionmode << cluster_mode_param;
    this->MakeSlotAvailable(&this->selectionmode);

    core::param::EnumParam* cluster_linkage_param =
        new core::param::EnumParam(static_cast<int>(LinkageMode::CENTROIDE));
    LinkageMode linkagemode = LinkageMode::CENTROIDE;
    cluster_linkage_param->SetTypePair(linkagemode, "Centroide Linkage");
    linkagemode = LinkageMode::SINGLE;
    cluster_linkage_param->SetTypePair(linkagemode, "Single Linkage");
    linkagemode = LinkageMode::AVARAGE;
    cluster_linkage_param->SetTypePair(linkagemode, "Avarage Linkage");
    this->linkagemodeparam << cluster_linkage_param;
    this->MakeSlotAvailable(&this->linkagemodeparam);

    core::param::EnumParam* cluster_moments_param = new core::param::EnumParam(static_cast<int>(MomentsMethode::IMAGE));
    MomentsMethode moments = MomentsMethode::IMAGE;
    cluster_moments_param->SetTypePair(moments, "Image-Moments");
    moments = MomentsMethode::COLOR;
    cluster_moments_param->SetTypePair(moments, "Color-Moments");
    moments = MomentsMethode::AIFEATURES;
    cluster_moments_param->SetTypePair(moments, "AI-Features");
    this->momentsmethode << cluster_moments_param;
    this->MakeSlotAvailable(&this->momentsmethode);

    this->distancemultiplier.SetParameter(new megamol::core::param::FloatParam(0.75, 0.0, 1.1));
    this->MakeSlotAvailable(&this->distancemultiplier);

    this->featuresSlot1Param.SetParameter(new megamol::core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->featuresSlot1Param);

    this->featuresSlot2Param.SetParameter(new megamol::core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->featuresSlot2Param);

    this->featuresSlot3Param.SetParameter(new megamol::core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->featuresSlot3Param);

    // Other Default Varialbles
    this->lastHash = 0;
    this->outHash = 0;
    this->outhashoffset = 0;

    this->datatocluster = false;
    this->dumpdotfile = false;
    this->selectionmodechanged = false;
    this->linkagemodechanged = false;
    this->distancemultiplierchanged = false;
    this->momentschanged = false;

    this->picturecount = 0;
}

/*
 * Clustering::~Clustering
 */
Clustering::~Clustering(void) {
    this->Release();
}

/*
 * Clustering::clusterData
 */
void Clustering::clusterData(
    image_calls::Image2DCall* cpp, image_calls::Image2DCall* cpp2, image_calls::Image2DCall* cpp3) {

    // if the last two are null, the data is already present and we can load it
    // otherwise, the data will be loaded during the HierarchicalClustering
    if (cpp2 == nullptr && cpp3 == nullptr) {
        this->picturecount = cpp->GetImageCount();
        this->fillPictureDataVector(*cpp);
    }

    if (this->momentsmethode.Param<core::param::EnumParam>()->Value() == MomentsMethode::AIFEATURES) {
        auto path1 = this->featuresSlot1Param.Param<core::param::FilePathParam>()->Value();
        auto path2 = this->featuresSlot2Param.Param<core::param::FilePathParam>()->Value();
        auto path3 = this->featuresSlot3Param.Param<core::param::FilePathParam>()->Value();

        this->loadFeatureVectorFromFile(path1, this->slot1Features);
        this->loadFeatureVectorFromFile(path2, this->slot2Features);
        this->loadFeatureVectorFromFile(path3, this->slot3Features);
    }

    // Clustering
    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "Clustering %I64u Pictures", this->picturecount);
    int mode = this->selectionmode.Param<core::param::EnumParam>()->Value();
    int bla = mode > 4 ? 1 : 2;
    mode = mode > 4 ? mode - 4 : mode;

    if (cpp2 == nullptr && cpp3 == nullptr) {
        this->clustering = new HierarchicalClustering(this->picdata, this->picturecount, this->slot1Features, this->distanceMatrix, mode, bla,
            this->linkagemodeparam.Param<core::param::EnumParam>()->Value(),
            this->momentsmethode.Param<core::param::EnumParam>()->Value());
    } else {
        this->clustering =
            new HierarchicalClustering(this->picdata, this->slot1Features, this->slot2Features, this->slot3Features, this->distanceMatrix,
                cpp, cpp2, cpp3, mode, bla, this->linkagemodeparam.Param<core::param::EnumParam>()->Value(),
                this->momentsmethode.Param<core::param::EnumParam>()->Value());
        this->picturecount = this->picdata.size();
    }
    core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO, "Clustering finished");
}

/*
 * Clustering::clusterData
 */
void Clustering::clusterData(CallClusteringLoader* ccl) {

    this->picturecount = ccl->Count();

    // Clustering
    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "Clustering %I64u Pictures", this->picturecount);
    this->clustering = new HierarchicalClustering(ccl->getLeaves(), this->distanceMatrix, ccl->Count(),
        this->momentsmethode.Param<core::param::EnumParam>()->Value(),
        this->selectionmode.Param<core::param::EnumParam>()->Value(),
        this->linkagemodeparam.Param<core::param::EnumParam>()->Value(),
        this->momentsmethode.Param<core::param::EnumParam>()->Value());
    core::utility::log::Log::DefaultLog.WriteMsg(
        core::utility::log::Log::LEVEL_INFO, "Clustering finished", this->picturecount);
}


/*
 * Clustering::create
 */
bool Clustering::create(void) {
    // intentionally empty
    return true;
}

/*
 * Clustering::getDataCallback
 */
bool Clustering::getDataCallback(core::Call& caller) {

    // Outgoing Call
    CallClustering* ccOut = dynamic_cast<CallClustering*>(&caller);
    if (ccOut == nullptr)
        return false;

    // Incoming Call
    image_calls::Image2DCall* imin = this->inSlotImageLoader.CallAs<image_calls::Image2DCall>();
    bool imageloader = (imin == nullptr);

    image_calls::Image2DCall* imin2 = this->inSlotImageLoader2.CallAs<image_calls::Image2DCall>();
    image_calls::Image2DCall* imin3 = this->inSlotImageLoader3.CallAs<image_calls::Image2DCall>();

    CallClusteringLoader* cclIn = this->inSlotCLUSTERINGLoader.CallAs<CallClusteringLoader>();
    bool clusterloader = (cclIn == nullptr);

    bool freshlyClustered = false;

    // Cluster Data
    if (this->datatocluster) {
        // reset new data flag
        this->datatocluster = false;

        if (!this->distanceFilePath.Param<core::param::FilePathParam>()->Value().empty()) {
            auto dpath = this->distanceFilePath.Param<core::param::FilePathParam>()->Value();
            DistanceMatrixLoader::load(dpath);
            this->distanceMatrix = DistanceMatrixLoader::getDistanceMap();
        }

        if (imageloader && clusterloader) {
            return false;
        } else {
            if (!imageloader) {

                if (imin2 == nullptr && imin3 == nullptr) {
                    if (!(*imin)(image_calls::Image2DCall::CallForWaitForData))
                        return false;
                    if (!(*imin)(image_calls::Image2DCall::CallForGetData))
                        return false;
                    if (!(*imin)(image_calls::Image2DCall::CallForWaitForData))
                        return false;
                    auto ptr = imin->GetImagePtr();
                    this->clusterData(imin);
                    freshlyClustered = true;
                } else {
                    this->clusterData(imin, imin2, imin3);
                    freshlyClustered = true;
                }
            }

            if (!clusterloader) {
                if (!(*cclIn)(CallClusteringLoader::CallForGetData))
                    return false;
                this->clusterData(cclIn);
                freshlyClustered = true;
            }
        }
    }

    // Dump-Dot File
    if (this->dumpdotfile) {
        this->dumpdotfile = false;

        const auto& filename = this->dumpdotpath.Param<core::param::FilePathParam>()->Value();
        if (this->clustering->finished()) {
            // Check filename for valid path
            if (!filename.empty()) {
                this->clustering->dump_dot(filename.c_str());
            } else {
                core::utility::log::Log::DefaultLog.WriteMsg(
                    core::utility::log::Log::LEVEL_INFO, "No Output-Filname given. File saved with defualt filename.");
                this->clustering->dump_dot();
            }
        }
    }

    // Recluster the Data because Mode changed
    if (this->selectionmodechanged) {
        this->selectionmodechanged = false;

        int value = this->selectionmode.Param<core::param::EnumParam>()->Value();
        switch (value) {
        case 1:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to Cosinus-Koeffizient");
            clustering->changeModeTo(2);
            clustering->setSimilarityMethod(value);
            break;
        case 2:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to Dice-Koeffizient");
            clustering->changeModeTo(2);
            clustering->setSimilarityMethod(value);
            break;
        case 3:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to Jaccard-Koeffizient");
            clustering->changeModeTo(2);
            clustering->setSimilarityMethod(value);
            break;
        case 4:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to Overlap-Koeffizient");
            clustering->changeModeTo(2);
            clustering->setSimilarityMethod(value);
            break;
        case 5:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to City-Block-Mannhatten");
            clustering->changeModeTo(1);
            clustering->setDistanceMethod(value - 4);
            break;
        case 6:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to Euclidian-Distance");
            clustering->changeModeTo(1);
            clustering->setDistanceMethod(value - 4);
            break;
        case 7:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to L3-Distance");
            clustering->changeModeTo(1);
            clustering->setDistanceMethod(value - 4);
            break;
        case 8:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Selection Mode changed to Gower Distance");
            clustering->changeModeTo(1);
            clustering->setDistanceMethod(value - 4);
            break;
        }

        // Recalculate clusterin
        if (!freshlyClustered)
            clustering->clusterthedata();
    }

    if (this->linkagemodechanged) {
        this->linkagemodechanged = false;

        int value = this->linkagemodeparam.Param<core::param::EnumParam>()->Value();
        switch (value) {
        case 1:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Linkage Mode changed to Centroid-Linkage");
            clustering->setLinkageMethod(1);
            break;
        case 2:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Linkage Mode changed to Single-Linkage");
            clustering->setLinkageMethod(2);
            break;
        case 3:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Linkage Mode changed to Avarage-Linkage");
            clustering->setLinkageMethod(3);
            break;
        }

        // Recalculate clusterin
        if (!freshlyClustered)
            clustering->clusterthedata();
    }

    if (this->distancemultiplierchanged) {
        float value = this->distancemultiplier.Param<core::param::FloatParam>()->Value();
        this->distancemultiplierchanged = false;
        this->clustering->setDistanceMultiplier(value);
    }

    if (this->momentschanged) {
        this->momentschanged = false;

        int value = this->momentsmethode.Param<core::param::EnumParam>()->Value();
        switch (value) {
        case 1:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Changed to Image-Moments");
            clustering->setMoments(1);
            break;
        case 2:
            core::utility::log::Log::DefaultLog.WriteMsg(
                core::utility::log::Log::LEVEL_INFO, "Changed to Color-Moments");
            clustering->setMoments(2);
            break;
        case 3:
            core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_INFO, "Changed to AI Features");
            clustering->setMoments(3);
        }

        // Reanalyse Pictures
        if (!freshlyClustered)
            clustering->reanalyse();
    }

    if (this->clustering->finished()) {
        if (ccOut->DataHash() != this->outHash) {
            // Send Clustering to CallClustering
            ccOut->setClustering(this->clustering);
        }
    }

    return true;
}

/*
 * Clustering::getExtentCallback
 */
bool Clustering::getExtentCallback(core::Call& caller) {

    // OUtgoing Call
    CallClustering* ccOut = dynamic_cast<CallClustering*>(&caller);
    if (ccOut == nullptr)
        return false;

    // Incoming Call
    image_calls::Image2DCall* cppIn = this->inSlotImageLoader.CallAs<image_calls::Image2DCall>();
    bool pngpicloader = (cppIn == nullptr);

    image_calls::Image2DCall* imin2 = this->inSlotImageLoader2.CallAs<image_calls::Image2DCall>();
    image_calls::Image2DCall* imin3 = this->inSlotImageLoader3.CallAs<image_calls::Image2DCall>();

    CallClusteringLoader* cclIn = this->inSlotCLUSTERINGLoader.CallAs<CallClusteringLoader>();
    bool clusterloader = (cclIn == nullptr);

    if (pngpicloader && clusterloader) {
        return false;
    } else {
        if (!pngpicloader) {
            if (!(*cppIn)(image_calls::Image2DCall::CallForGetMetaData))
                return false;

            bool bothNull = true;
            if (imin2 != nullptr) {
                if (!(*imin2)(image_calls::Image2DCall::CallForGetMetaData))
                    return false;
                bothNull = false;
            }
            if (imin3 != nullptr) {
                if (!(*imin3)(image_calls::Image2DCall::CallForGetMetaData))
                    return false;
                bothNull = false;
            }
            if (bothNull) {
                if (!(*cppIn)(image_calls::Image2DCall::CallForSetWishlist))
                    return false;
            }
        }

        if (!clusterloader) {
            if (!(*cclIn)(CallClusteringLoader::CallForGetExtent))
                return false;
        }
    }

    // Check for new Data
    if (!pngpicloader) {
        if (lastHash != cppIn->DataHash()) {
            lastHash = cppIn->DataHash();
            this->datatocluster = true;
            this->outhashoffset++;
        }
    } else if (!clusterloader) {
        if (lastHash != cclIn->DataHash()) {
            lastHash = cclIn->DataHash();
            this->datatocluster = true;
            this->outhashoffset++;
        }
    }

    if (this->dumpdot.IsDirty()) {
        this->dumpdotfile = true;
        this->dumpdot.ResetDirty();
    }
    bool hashchange = false;

    if (this->selectionmode.IsDirty()) {
        this->selectionmodechanged = true;
        this->selectionmode.ResetDirty();
        hashchange = true;
    }
    if (this->linkagemodeparam.IsDirty()) {
        this->linkagemodechanged = true;
        this->linkagemodeparam.ResetDirty();
        hashchange = true;
    }
    if (this->distancemultiplier.IsDirty()) {
        this->distancemultiplierchanged = true;
        this->distancemultiplier.ResetDirty();
        hashchange = true;
    }
    if (this->momentsmethode.IsDirty()) {
        this->momentschanged = true;
        this->momentsmethode.ResetDirty();
        hashchange = true;
    }

    if (hashchange)
        this->outhashoffset++;

    if (ccOut->DataHash() != this->outHash + this->outhashoffset)
        ccOut->SetDataHash(this->outHash + this->outhashoffset);
    return true;
}

/*
 * Clustering::release
 */
void Clustering::release(void) {}

void Clustering::fillPictureDataVector(image_calls::Image2DCall& imc) {
    auto imcount = imc.GetImagePtr()->size();
    this->picdata.clear();
    this->picdata.resize(imcount);
    uint32_t id = 0;
    for (auto& p : *imc.GetImagePtr()) {
        this->picdata[id].width = p.second.Width();
        this->picdata[id].height = p.second.Height();
        this->picdata[id].path = p.first;
        this->picdata[id].pdbid = std::filesystem::path(p.first).stem().string();
        this->picdata[id].render = false;
        this->picdata[id].popup = false;
        this->picdata[id].texture = nullptr;
        this->picdata[id].image = &p.second;
        this->loadValueImageForGivenPicture(std::filesystem::path(p.first), this->picdata[id].valueImage);
        auto minmax = std::minmax_element(this->picdata[id].valueImage.begin(), this->picdata[id].valueImage.end());
        this->picdata[id].minValue = (*minmax.first);
        this->picdata[id].maxValue = (*minmax.second);
        ++id;
    }
}

void Clustering::loadValueImageForGivenPicture(
    const std::filesystem::path& originalPicture, std::vector<float>& outValueImage) {
    auto newpath = originalPicture;
    newpath = newpath.parent_path();
    newpath.append(originalPicture.stem().string() + "_values.dat");
    newpath = newpath.make_preferred();

    auto filesize = std::filesystem::file_size(newpath);
    std::ifstream file(newpath, std::ios::binary);
    outValueImage.clear();
    outValueImage.resize(filesize / sizeof(float));
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(outValueImage.data()), filesize);
        file.close();
        auto minmax = std::minmax_element(outValueImage.begin(), outValueImage.end());
        auto minele = std::abs(*minmax.first);
        for (auto& v : outValueImage) {
            v += minele;
            v = std::abs(v); // paranoia
        }
    } else {
        core::utility::log::Log::DefaultLog.WriteError(
            "The file \"%s\" could not be opened for reading", newpath.c_str());
    }
}

bool Clustering::loadFeatureVectorFromFile(
    const std::filesystem::path& filePath, std::map<std::string, std::vector<float>>& outFeatureMap) {
    auto fsize = std::filesystem::file_size(filePath);
    std::vector<FeatureStruct> readVec;

    std::ifstream file(filePath, std::ios::binary);
    if (file.is_open()) {
        int ver;
        file.read(reinterpret_cast<char*>(&ver), sizeof(int));
        if (ver != 0) {
            file.seekg(0, std::ios::beg);
            auto numstructs = fsize / sizeof(FeatureStruct);
            readVec.resize(numstructs);
            file.read(reinterpret_cast<char*>(readVec.data()), fsize);
            file.close();

            outFeatureMap.clear();
            for (const auto& el : readVec) {
                std::string id;
                id.append(el.pdbId.data(), 4);
                std::vector<float> copy(el.featureVec.begin(), el.featureVec.end());
                outFeatureMap.insert(std::make_pair(id, copy));
            }
        } else {
            int length = 0;
            file.read(reinterpret_cast<char*>(&length), sizeof(int));
            std::vector<char> pdbId(length);
            pdbId.shrink_to_fit();
            DataStruct str;
            uint64_t numstructs = fsize / (length + (sizeof(float) * 1792));
            outFeatureMap.clear();
            for (uint64_t i = 0; i < numstructs; ++i) {
                file.read(reinterpret_cast<char*>(pdbId.data()), length);
                file.read(reinterpret_cast<char*>(str.featureVec.data()), sizeof(float) * str.featureVec.size());
                str.pdbId.clear();
                str.pdbId.append(pdbId.data(), length);
                // remove spaces
                str.pdbId.erase(std::remove_if(str.pdbId.begin(), str.pdbId.end(),
                                    [](char& c) { return std::isspace<char>(c, std::locale::classic()); }),
                    str.pdbId.end());
                std::vector<float> copy(str.featureVec.begin(), str.featureVec.end());
                outFeatureMap.insert(std::make_pair(str.pdbId, copy));
            }
        }
    } else {
        return false;
    }
    return true;
}
