#include "Clustering_2.h"

#include "CallClustering_2.h"
#include "image_calls/Image2DCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"

#include <filesystem>
#include <fstream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster_gl;

Clustering_2::Clustering_2(void)
        : Module()
        , getImageSlot("getImage", "Slot to retrieve the images to cluster over")
        , getImageSlot2("getImage2", "Slot to retrieve additional images to cluster over")
        , getImageSlot3("getImage3", "Slot to retrieve additional images to cluster over")
        , getImageSlot4("getImage4", "Slot to retrieve additional images to cluster over")
        , sendClusterSlot("sendCluster", "Slot to send the resulting clustering")
        , image_1_features_slot_("features::image1", "Feature file path for the first image call")
        , image_2_features_slot_("features::image2", "Feature file path for the second image call")
        , image_3_features_slot_("features::image3", "Feature file path for the third image call")
        , image_4_features_slot_("features::image4", "Feature file path for the fourth image call")
        , useMultipleMapsParam("useMultipleMaps", "Enables that the clustering uses multiple maps instead of one")
        , clusteringMethodSelectionParam("mode::clusteringMethod", "Selection of the used clustering method")
        , distanceMeasureSelectionParam("mode::distanceMeasure", "Selection of the used distance measure")
        , dataHashOffset(0) {
    // Callee slots
    this->sendClusterSlot.SetCallback(CallClustering_2::ClassName(),
        CallClustering_2::FunctionName(CallClustering_2::CallForGetExtent), &Clustering_2::GetExtentCallback);
    this->sendClusterSlot.SetCallback(CallClustering_2::ClassName(),
        CallClustering_2::FunctionName(CallClustering_2::CallForGetData), &Clustering_2::GetDataCallback);
    this->MakeSlotAvailable(&this->sendClusterSlot);

    // Caller slots
    this->getImageSlot.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot);

    this->getImageSlot2.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot2);

    this->getImageSlot3.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot3);

    this->getImageSlot4.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->getImageSlot4);

    // Param slots
    this->useMultipleMapsParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->useMultipleMapsParam);

    image_1_features_slot_.SetParameter(
        new core::param::FilePathParam("", param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&image_1_features_slot_);

    image_2_features_slot_.SetParameter(
        new core::param::FilePathParam("", param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&image_2_features_slot_);

    image_3_features_slot_.SetParameter(
        new core::param::FilePathParam("", param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&image_3_features_slot_);

    image_4_features_slot_.SetParameter(
        new core::param::FilePathParam("", param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&image_4_features_slot_);

    auto* clusteringMethodEnum = new core::param::EnumParam(static_cast<int>(ClusteringMethod::FILEFEATURES));
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::IMAGE_MOMENTS), "Image Moments");
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::COLOR_MOMENTS), "Color Moments");
    clusteringMethodEnum->SetTypePair(static_cast<int>(ClusteringMethod::FILEFEATURES), "Features from File");
    this->clusteringMethodSelectionParam.SetParameter(clusteringMethodEnum);
    this->MakeSlotAvailable(&this->clusteringMethodSelectionParam);

    auto* distanceMeasureEnum = new core::param::EnumParam(static_cast<int>(DistanceMeasure::EUCLIDEAN_DISTANCE));
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::EUCLIDEAN_DISTANCE), "Euclidean Distance");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::L3_DISTANCE), "L3 Distance");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::COSINUS_DISTANCE), "Cosine Similarity");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::DICE_DISTANCE), "Dice Similarity");
    distanceMeasureEnum->SetTypePair(static_cast<int>(DistanceMeasure::JACCARD_DISTANCE), "Jaccard Similarity");
    this->distanceMeasureSelectionParam.SetParameter(distanceMeasureEnum);
    this->MakeSlotAvailable(&this->distanceMeasureSelectionParam);

    // Variables
    this->nodes = std::make_shared<std::vector<ClusterNode_2>>();
    this->lastDataHash = {0, 0, 0, 0};
}

Clustering_2::~Clustering_2(void) {
    this->Release();
}

bool Clustering_2::create(void) {
    // TODO
    return true;
}

void Clustering_2::release(void) {
    // TODO
}

bool Clustering_2::GetDataCallback(Call& call) {
    auto cc = dynamic_cast<CallClustering_2*>(&call);
    if (cc == nullptr)
        return false;

    if (this->recalculateClustering) {
        this->recalculateClustering = false;
        if (!runComputation())
            return false;
    }

    // TODO copy data

    return true;
}

bool Clustering_2::GetExtentCallback(Call& call) {
    CallClustering_2* cc = dynamic_cast<CallClustering_2*>(&call);
    if (cc == nullptr)
        return false;

    if (this->checkParameterDirtyness()) {
        this->recalculateClustering = true;
    }

    std::vector<std::pair<image_calls::Image2DCall*, int>> calls;
    {
        auto imc1 = this->getImageSlot.CallAs<image_calls::Image2DCall>();
        auto imc2 = this->getImageSlot2.CallAs<image_calls::Image2DCall>();
        auto imc3 = this->getImageSlot3.CallAs<image_calls::Image2DCall>();
        auto imc4 = this->getImageSlot4.CallAs<image_calls::Image2DCall>();

        if (imc1 != nullptr)
            calls.push_back(std::make_pair(imc1, 0));
        if (imc2 != nullptr)
            calls.push_back(std::make_pair(imc2, 1));
        if (imc3 != nullptr)
            calls.push_back(std::make_pair(imc3, 2));
        if (imc4 != nullptr)
            calls.push_back(std::make_pair(imc4, 3));
    }
    // if the first one is not connected, fail
    if (calls.empty() || calls.at(0).first == nullptr)
        return false;

    // get the metadata for all relevant calls
    if (this->useMultipleMapsParam.Param<param::BoolParam>()->Value()) {
        for (const auto& c : calls) {
            if (!(*c.first)(image_calls::Image2DCall::CallForGetMetaData))
                return false;
            if (this->lastDataHash.at(c.second) != c.first->DataHash()) {
                this->lastDataHash.at(c.second) = c.first->DataHash();
                this->recalculateClustering = true;
            }
        }
    } else {
        if (!(*calls.at(0).first)(image_calls::Image2DCall::CallForGetMetaData))
            return false;
        if (this->lastDataHash.at(calls.at(0).second) != calls.at(0).first->DataHash()) {
            this->lastDataHash.at(calls.at(0).second) = calls.at(0).first->DataHash();
            this->recalculateClustering = true;
        }
    }

    if (this->recalculateClustering) {
        this->dataHashOffset++;
    }

    ClusteringMetaData meta;
    meta.dataHash = this->dataHashOffset;
    for (const auto& val : this->lastDataHash) {
        meta.dataHash += val;
    }
    //meta.numLeafNodes = calls.at(0).first->GetImageCount();
    meta.numLeafNodes = calls.at(0).first->GetAvailablePathsCount();
    cc->SetMetaData(meta);

    return true;
}

bool Clustering_2::checkParameterDirtyness(void) {
    bool result = false;
    if (this->clusteringMethodSelectionParam.IsDirty()) {
        this->clusteringMethodSelectionParam.ResetDirty();
        result = true;
    }
    if (this->distanceMeasureSelectionParam.IsDirty()) {
        this->distanceMeasureSelectionParam.ResetDirty();
        result = true;
    }
    if (this->useMultipleMapsParam.IsDirty()) {
        this->useMultipleMapsParam.ResetDirty();
        result = true;
    }
    if (image_1_features_slot_.IsDirty()) {
        image_1_features_slot_.ResetDirty();
        result = true;
    }
    if (image_2_features_slot_.IsDirty()) {
        image_2_features_slot_.ResetDirty();
        result = true;
    }
    if (image_3_features_slot_.IsDirty()) {
        image_3_features_slot_.ResetDirty();
        result = true;
    }
    if (image_4_features_slot_.IsDirty()) {
        image_4_features_slot_.ResetDirty();
        result = true;
    }
    return result;
}

bool Clustering_2::loadFeatureVectorFromFile(
    const std::filesystem::path& file_path, std::map<std::string, std::vector<float>>& OUT_feature_map) const {

    auto fsize = std::filesystem::file_size(file_path);
    std::vector<FeatureData> readVec;

    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        int ver;
        file.read(reinterpret_cast<char*>(&ver), sizeof(int));
        uint64_t numstructs = 0;
        int length = 4;
        if (ver != 0) {
            file.seekg(0, std::ios::beg);
        } else {
            file.read(reinterpret_cast<char*>(&length), sizeof(int));
        }
        // TODO currently, the feature vector size of 1792 is hardcoded, that should change
        constexpr int veclen = 1792;
        numstructs = fsize / (length + (sizeof(float) * veclen));
        std::vector<char> pdbId(length);
        pdbId.shrink_to_fit();
        FeatureData str;
        str.feature_vec.resize(veclen);
        OUT_feature_map.clear();
        for (uint64_t i = 0; i < numstructs; ++i) {
            file.read(reinterpret_cast<char*>(pdbId.data()), length);
            file.read(reinterpret_cast<char*>(str.feature_vec.data()), sizeof(float) * str.feature_vec.size());
            str.pdb_id.clear();
            str.pdb_id.append(pdbId.data(), length);
            // remove spaces
            str.pdb_id.erase(std::remove_if(str.pdb_id.begin(), str.pdb_id.end(),
                                 [](char& c) { return std::isspace<char>(c, std::locale::classic()); }),
                str.pdb_id.end());
            //std::vector<float> copy(str.feature_vec.begin(), str.feature_vec.end());
            OUT_feature_map.insert(std::make_pair(str.pdb_id, str.feature_vec));
        }
    } else {
        return false;
    }
    return true;
}


bool Clustering_2::runComputation(void) {
    if (!this->calculateFeatureVectors()) {
        core::utility::log::Log::DefaultLog.WriteError("[Clustering_2]: Feature Vector calculation failed!");
        return false;
    }
    if (!this->clusterImages()) {
        core::utility::log::Log::DefaultLog.WriteError("[Clustering_2]: Clustering calculation failed!");
        return false;
    }
    return true;
}


bool Clustering_2::calculateFeatureVectors(void) {
    std::vector<std::pair<image_calls::Image2DCall*, int>> calls;
    {
        auto imc1 = this->getImageSlot.CallAs<image_calls::Image2DCall>();
        auto imc2 = this->getImageSlot2.CallAs<image_calls::Image2DCall>();
        auto imc3 = this->getImageSlot3.CallAs<image_calls::Image2DCall>();
        auto imc4 = this->getImageSlot4.CallAs<image_calls::Image2DCall>();

        if (imc1 != nullptr)
            calls.push_back(std::make_pair(imc1, 0));
        if (imc2 != nullptr)
            calls.push_back(std::make_pair(imc2, 1));
        if (imc3 != nullptr)
            calls.push_back(std::make_pair(imc3, 2));
        if (imc4 != nullptr)
            calls.push_back(std::make_pair(imc4, 3));
    }
    // if nothing is connected, fail
    if (calls.empty())
        return false;

    if (!useMultipleMapsParam.Param<param::BoolParam>()->Value() && calls.size() > 1) {
        while (calls.size() > 1) {
            calls.pop_back();
        }
    }

    // get the metadata for all relevant calls
    std::vector<size_t> call_sizes;
    for (const auto& c : calls) {
        if (!(*c.first)(image_calls::Image2DCall::CallForGetMetaData))
            return false;
        call_sizes.emplace_back((*c.first).GetAvailablePathsCount());
    }
    auto const minmax = std::minmax_element(call_sizes.begin(), call_sizes.end());
    if (*minmax.first != *minmax.second) // each needs the same amount of images
        return false;

    auto const method =
        static_cast<ClusteringMethod>(clusteringMethodSelectionParam.Param<param::EnumParam>()->Value());

    switch (method) {
    case ClusteringMethod::COLOR_MOMENTS:
    case ClusteringMethod::IMAGE_MOMENTS:
        return calculateMomentsFeatureVectors(calls, method);
    case ClusteringMethod::FILEFEATURES:
        return calculateFileFeatureVectors(calls);
    }
    return false;
}

bool Clustering_2::calculateMomentsFeatureVectors(
    std::vector<std::pair<image_calls::Image2DCall*, int>> const& calls, ClusteringMethod method) {
    if (method != ClusteringMethod::COLOR_MOMENTS && method != ClusteringMethod::IMAGE_MOMENTS) {
        return false;
    }
    feature_vectors_.clear();
    for (auto& cur : calls) {
        auto& call = cur.first;
        auto const& paths = call->GetAvailablePathsPtr();

        if (paths != nullptr) {
            for (auto const& path : *paths) {
                std::filesystem::path p_path(path);
                std::string const pdbid = p_path.stem().string();
                auto val_image_path = valumeImageNameFromNormalImage(path);
                std::vector<float> val_image;
                if (!loadValueImageFromFile(val_image_path, val_image)) {
                    feature_vectors_.clear();
                    return false;
                }
                std::vector<float> feat_vec;
                if (method == ClusteringMethod::COLOR_MOMENTS) {
                    calcColorMomentsForValueImage(val_image, feat_vec);
                }
                if (method == ClusteringMethod::IMAGE_MOMENTS) {
                    calcImageMomentsForValueImage(val_image, feat_vec);
                }
                if (feature_vectors_.count(pdbid) == 0) {
                    feature_vectors_[pdbid] = feat_vec;
                } else {
                    feature_vectors_[pdbid].insert(feature_vectors_[pdbid].end(), feat_vec.begin(), feat_vec.end());
                }
            }
        }
    }
    return true;
}

bool Clustering_2::calculateFileFeatureVectors(std::vector<std::pair<image_calls::Image2DCall*, int>> const& calls) {
    feature_vectors_.clear();
    for (auto& cur : calls) {
        auto& call = cur.first;
        auto const& paths = call->GetAvailablePathsPtr();
        auto const feature_file = getPathForIndex(cur.second);
        std::map<std::string, std::vector<float>> feature_map_from_file;

        if (!loadFeatureVectorFromFile(feature_file, feature_map_from_file)) {
            utility::log::Log::DefaultLog.WriteError(
                "[Clustering_2]: The feature file \"%s\" could not be loaded", feature_file.c_str());
            feature_vectors_.clear();
            return false;
        }

        if (paths != nullptr) {
            for (auto const& path : *paths) {
                std::filesystem::path p_path(path);
                std::string const pdbid = p_path.stem().string();
                if (feature_map_from_file.count(pdbid) == 0) {
                    utility::log::Log::DefaultLog.WriteError(
                        "[Clustering_2]: The features for the PDB-ID \"%s\" could not be found in the file \"%s\"",
                        pdbid.c_str(), feature_file.c_str());
                    feature_vectors_.clear();
                    return false;
                }
                auto feat_vec = feature_map_from_file[pdbid];
                if (feature_vectors_.count(pdbid) == 0) {
                    feature_vectors_[pdbid] = feat_vec;
                } else {
                    feature_vectors_[pdbid].insert(feature_vectors_[pdbid].end(), feat_vec.begin(), feat_vec.end());
                }
            }
        }
    }
    return true;
}

bool Clustering_2::clusterImages(void) {
    // TODO implement
    return true;
}

std::string Clustering_2::valumeImageNameFromNormalImage(const std::string& str) const {
    if (str.empty()) {
        return "";
    }
    std::filesystem::path fres(str);
    if (!is_regular_file(fres)) {
        return "";
    }
    auto stem = fres.stem().string();
    stem += "_values";
    fres.replace_filename(stem);
    fres.replace_extension(".dat");
    return fres.string();
}

std::filesystem::path Clustering_2::getPathForIndex(int const index) {
    std::filesystem::path result;
    switch (index) {
    case 0:
        result = image_1_features_slot_.Param<param::FilePathParam>()->Value();
        break;
    case 1:
        result = image_2_features_slot_.Param<param::FilePathParam>()->Value();
        break;
    case 2:
        result = image_3_features_slot_.Param<param::FilePathParam>()->Value();
        break;
    case 3:
        result = image_4_features_slot_.Param<param::FilePathParam>()->Value();
        break;
    default:
        break;
    }
    return result;
}

bool Clustering_2::loadValueImageFromFile(
    const std::filesystem::path& file_path, std::vector<float>& OUT_value_image, bool normalize_values) const {
    OUT_value_image.clear();
    if (!std::filesystem::is_regular_file(file_path)) {
        utility::log::Log::DefaultLog.WriteError(
            "[Clustering_2]: the value file \"%s\" is no regular file", file_path.c_str());
        return false;
    }
    auto const file_size = std::filesystem::file_size(file_path);
    OUT_value_image.resize(file_size / sizeof(float));
    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(OUT_value_image.data()), file_size);
        file.close();
        if (normalize_values) {
            auto const minmax = std::minmax_element(OUT_value_image.begin(), OUT_value_image.end());
            auto const minele = *minmax.first;
            for (auto& v : OUT_value_image) {
                v -= minele;
                v = std::abs(v); // paranoia
            }
        }
    } else {
        utility::log::Log::DefaultLog.WriteError(
            "[Clustering_2]: the value file \"%s\" could not be opened", file_path.c_str());
        return false;
    }
    return true;
}

void Clustering_2::calcColorMomentsForValueImage(
    std::vector<float> const& val_image, std::vector<float>& OUT_feature_vector) const {

    double mean = 0.0;
    double deviation = 0.0;
    double skewness = 0.0;

    // TODO find way to remove the hardcoded image measurements
    auto constexpr factor = static_cast<double>(picwidth * picheight);

    // Calculate Mean
    double summean = 0.0;
    for (int i = 0; i < picheight; ++i) {
        for (int j = 0; j < picwidth; ++j) {
            summean += static_cast<double>(val_image[picwidth * i + j]);
        }
    }

    mean = summean / factor;

    // Calculate Deviation and Skewness
    double deviationsum = 0.0;
    double skewnesssum = 0.0;
    for (int i = 0; i < picheight; ++i) {
        for (int j = 0; j < picwidth; ++j) {
            deviationsum += std::pow(static_cast<double>(val_image[picwidth * i + j]) - mean, 2);
            skewnesssum += std::pow(static_cast<double>(val_image[picwidth * i + j]) - mean, 3);
        }
    }

    // Calculate Deviation
    deviation = std::sqrt(deviationsum / factor);
    skewness = std::cbrt((skewnesssum / factor));

    OUT_feature_vector = {static_cast<float>(mean), static_cast<float>(deviation), static_cast<float>(skewness)};
}

void Clustering_2::calcImageMomentsForValueImage(
    std::vector<float> const& val_image, std::vector<float>& OUT_feature_vector) const {

    double m00 = 0.0;
    double m01 = 0.0;
    double m10 = 0.0;

    // TODO find way to remove the hardcoded image measurements
    constexpr int order = 3;

    for (int y = 0; y < picheight; y++) {
        for (int x = 0; x < picwidth; x++) {
            m00 += static_cast<double>(val_image[(y * picwidth) + x]);
            m10 += static_cast<double>(x) * static_cast<double>(val_image[(y * picwidth) + x]);
            m01 += static_cast<double>(y) * static_cast<double>(val_image[(y * picwidth) + x]);
        }
    }

    double const xc = m10 / m00;
    double const yc = m01 / m00;

    double müij[order + 1][order + 1] = {0.0};
    for (int i = 0; i <= order; i++) {
        for (int j = 0; j <= order; j++) {
            müij[i][j] = 0.0;
            for (int y = 0; y < picheight; y++) {
                for (int x = 0; x < picwidth; x++) {
                    if ((i + j <= 3) && !((i == 1 && j == 0) || (i == 0 && j == 1))) {
                        müij[i][j] += pow(static_cast<double>(x) - xc, i) * pow(static_cast<double>(y) - yc, j) *
                                      static_cast<double>(val_image[(y * picwidth) + x]);
                    }
                }
            }
        }
    }

    std::vector<double> nu;
    for (int i = 0; i <= order; i++) {
        for (int j = 0; j <= order; j++) {
            nu.push_back(müij[i][j] / (pow(müij[0][0], (1.0 + (static_cast<double>(i + j) / 2.0)))));
        }
    }

    double const i1 = nu[2 * order + 0] + nu[2];
    double const i2 = pow(nu[2 * order + 0] - nu[2], 2) + 4.0 * pow(nu[1 * order + 1], 2);
    // double i3 =
    //    pow(nu[3 * order + 0] - 3.0 * nu[1 * order + 2], 2) + pow(3.0 * nu[2 * order + 1] - nu[3],
    //    2);
    double const i4 = pow(nu[3 * order + 0] + nu[1 * order + 2], 2) + pow(nu[2 * order + 1] + nu[3], 2);
    double const i5 = (nu[3 * order + 0] - 3.0 * nu[1 * order + 2]) * (nu[3 * order + 0] + nu[1 * order + 2]) *
                          (pow((nu[3 * order + 0] + nu[1 * order + 2]), 2) - 3.0 * pow(nu[2 * order + 1] + nu[3], 2)) +
                      (3 * nu[2 * order + 1] - nu[3]) * (nu[2 * order + 1] + nu[3]) *
                          (3.0 * pow(nu[3 * order + 0] + nu[1 * order + 2], 2) - pow(nu[2 * order + 1] + nu[3], 2));
    double const i6 = (nu[2 * order + 0] - nu[2]) *
                          (pow(nu[3 * order + 0] + nu[1 * order + 2], 2) - pow(nu[2 * order + 1] + nu[3], 2)) +
                      4.0 * nu[1 * order + 1] * (nu[3 * order + 0] + nu[1 * order + 2]) * (nu[2 * order + 1] + nu[3]);
    double const i7 = (3.0 * nu[2 * order + 1] - nu[3]) * (nu[3 * order + 0] + nu[1 * order + 2]) *
                          (pow((nu[3 * order + 0] + nu[1 * order + 2]), 2) - 3.0 * pow(nu[2 * order + 1] + nu[3], 2)) -
                      (nu[3 * order + 0] - 3 * nu[1 * order + 2]) * (nu[2 * order + 1] + nu[3]) *
                          (3.0 * pow(nu[3 * order + 0] + nu[1 * order + 2], 2) - pow(nu[2 * order + 1] + nu[3], 2));
    double const i8 =
        nu[1 * order + 1] * (pow(nu[3 * order + 0] + nu[1 * order + 2], 2) - pow(nu[3] + nu[2 * order + 1], 2)) -
        (nu[2 * order + 0] - nu[2]) * (nu[3 * order + 0] + nu[1 * order + 2]) * (nu[3] + nu[2 * order + 1]);

    OUT_feature_vector = {static_cast<float>(i1), 0.0, 0.0, static_cast<float>(i2), static_cast<float>(i4),
        static_cast<float>(i5), static_cast<float>(i6), static_cast<float>(i7), static_cast<float>(i8)};
}
