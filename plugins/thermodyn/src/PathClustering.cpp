#include "stdafx.h"
#include "PathClustering.h"

#include <variant>

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmstd_datatools/DBSCAN.h"
#include "thermodyn/PathLineDataCall.h"
#include "mmcore/param/BoolParam.h"


megamol::thermodyn::PathClustering::PathClustering()
    : dataInSlot_("dataIn", "Input of particle pathlines")
    , dataOutSlot_("dataOut", "Output of clustered pathlines")
    , minPtsSlot_("minPts", "MinPts param for DBSCAN")
    , sigmaSlot_("sigma", "Sigma param for DBSCAN")
    , conDirsSlot_("use_dirs", "Use dirs with clustering")
    , conTempSlot_("use_temp", "Use temp with clustering") { 
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &PathClustering::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &PathClustering::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);

    minPtsSlot_ << new core::param::IntParam(1, 1, std::numeric_limits<int>::max());
    MakeSlotAvailable(&minPtsSlot_);

    sigmaSlot_ << new core::param::FloatParam(
        0.5f, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    MakeSlotAvailable(&sigmaSlot_);

    conDirsSlot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&conDirsSlot_);

    conTempSlot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&conTempSlot_);
}


megamol::thermodyn::PathClustering::~PathClustering() { this->Release(); }


bool megamol::thermodyn::PathClustering::create() { return true; }


void megamol::thermodyn::PathClustering::release() {}


bool megamol::thermodyn::PathClustering::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_ || isDirty()) {
        inDataHash_ = inCall->DataHash();
        resetDirty();

        vislib::sys::Log::DefaultLog.WriteInfo("PathClustering: Calculating clustering\n");

        auto const minPts = minPtsSlot_.Param<core::param::IntParam>()->Value();
        auto const sigma = sigmaSlot_.Param<core::param::FloatParam>()->Value();

        auto const pathStore = inCall->GetPathStore();
        auto const& entrySizes = inCall->GetEntrySize();
        auto const& inDirsPresent = inCall->HasDirections();
        auto const& inColsPresent = inCall->HasColors();

        auto const frameCount = inCall->GetTimeSteps();

        auto const bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();

        outEntrySizes_.resize(pathStore->size(), 3);
        outDirsPresent_.resize(pathStore->size());
        outColsPresent_.resize(pathStore->size());
        outPathStore_.resize(pathStore->size());

        auto conDirs = conDirsSlot_.Param<core::param::BoolParam>()->Value();
        auto const conTemp = conTempSlot_.Param<core::param::BoolParam>()->Value();
        if (conTemp) conDirs = false;

        for (size_t plidx = 0; plidx < pathStore->size(); ++plidx) {
            auto const& paths = pathStore->operator[](plidx);
            if (paths.empty()) continue;

            auto const entrySize = entrySizes[plidx];
            size_t tempOffset = 3;
            if (inColsPresent[plidx]) tempOffset += 4;
            if (inDirsPresent[plidx]) tempOffset += 3;

            vertex_set_t vertexSet;
            vertexSet.reserve(frameCount * 100);
            /*edge_set_t edgeSet;
            edgeSet.reserve(frameCount*100);*/

            std::vector<float> db_input;
            db_input.reserve(paths.size() * frameCount * 5);

#define DBSCAN_DIM 4

            std::variant<DB_3DIM, DB_4DIM, DB_7DIM> db;
            std::vector<std::vector<float>> clusters;

            // initialize ds with frame 0
            prepareFrameData(db_input, paths, 0, entrySize, tempOffset, inColsPresent[plidx], inDirsPresent[plidx] && conDirs, conTemp);
            //stdplugin::datatools::DBSCAN<float, true, DBSCAN_DIM, true>::cluster_set_t clusters;
            {
                if (!conTemp) {
                    db = DB_3DIM(paths.size(), 4, db_input, bbox, minPts, sigma);
                    clusters = std::get<DB_3DIM>(db).Scan();
                } else if (inDirsPresent[plidx] && conDirs && conTemp) {
                    db = DB_7DIM(paths.size(), 8, db_input, bbox, minPts, sigma);
                    clusters = std::get<DB_7DIM>(db).Scan();
                } else {
                    db = DB_4DIM(paths.size(), 5, db_input, bbox, minPts, sigma);
                    clusters = std::get<DB_4DIM>(db).Scan();
                }
            }

            /*std::vector<size_t> assigned;
            assigned.reserve(paths.size());*/
            cluster_assoc_t cluster_assoc;
            createClusterAssoc(0, clusters, cluster_assoc, inDirsPresent[plidx] && conDirs, conTemp);
            auto localVset = getVertexList(clusters, inDirsPresent[plidx] && conDirs, conTemp);
            vertexSet.insert(vertexSet.end(), localVset.begin(), localVset.end());

            for (size_t fidx = 1; fidx < frameCount; ++fidx) {
                // progressClusterAssoc(cluster_assoc, assigned);
                // assigned.clear();
                // continue with the rest of the frames
                db_input.clear();
                prepareFrameData(db_input, paths, fidx, entrySize, tempOffset, inColsPresent[plidx],
                    inDirsPresent[plidx] && conDirs, conTemp);
                {
                    if (!conTemp) {
                        db = DB_3DIM(paths.size(), 4, db_input, bbox, minPts, sigma);
                        clusters = std::get<DB_3DIM>(db).Scan();
                    } else if (inDirsPresent[plidx] && conDirs) {
                        db = DB_7DIM(paths.size(), 8, db_input, bbox, minPts, sigma);
                        clusters = std::get<DB_7DIM>(db).Scan();
                    } else {
                        db = DB_4DIM(paths.size(), 5, db_input, bbox, minPts, sigma);
                        clusters = std::get<DB_4DIM>(db).Scan();
                    }
                }
                createClusterAssoc(vertexSet.size(), clusters, cluster_assoc, inDirsPresent[plidx] && conDirs, conTemp);
                auto localVset = getVertexList(clusters, inDirsPresent[plidx] && conDirs, conTemp);
                // auto localEset = getEdgeList(cluster_assoc);
                vertexSet.insert(vertexSet.end(), localVset.begin(), localVset.end());
                // edgeSet.insert(edgeSet.end(), localEset.begin(), localEset.end());
            }

            auto& outPaths = outPathStore_[plidx];
            outPaths.clear();
            outPaths.reserve(cluster_assoc.size());
            // create pathlines from the set of vertices and edges
            for (auto const& el : cluster_assoc) {
                // each entry in cluster_assoc is a pathline
                auto pathline = createPathline(el.second, vertexSet);
                outPaths[el.first] = pathline;
            }
        }
    }

    outCall->SetPathStore(&outPathStore_);
    outCall->SetColorFlags(outColsPresent_);
    outCall->SetDirFlags(outDirsPresent_);
    outCall->SetEntrySizes(outEntrySizes_);
    outCall->SetTimeSteps(inCall->GetTimeSteps());

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


bool megamol::thermodyn::PathClustering::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(1)) return false;


    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    /*outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox_);
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox_);*/
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);


    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);

    outCall->SetDataHash(inDataHash_);

    return true;
}
