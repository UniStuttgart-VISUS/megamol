#include "stdafx.h"
#include "PathPCA.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/IntParam.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathPCA::PathPCA()
    : dataInSlot_("dataIn", "Input of particle pathlines")
    , dataOutSlot_("dataOut", "Output of PCA")
    , numFeatureSlot_("numFeatures", "Number of PCA features") {
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    /*dataOutSlot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PathPCA::getDataCallback);
    dataOutSlot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PathPCA::getExtentCallback);*/
    dataOutSlot_.SetCallback(PathLineDataCall::ClassName(),
        PathLineDataCall::FunctionName(0), &PathPCA::getDataCallback);
    dataOutSlot_.SetCallback(PathLineDataCall::ClassName(),
        PathLineDataCall::FunctionName(1), &PathPCA::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);

    numFeatureSlot_ << new core::param::IntParam(3, 1, std::numeric_limits<int>::max());
    MakeSlotAvailable(&numFeatureSlot_);
}


megamol::thermodyn::PathPCA::~PathPCA() { this->Release(); }


bool megamol::thermodyn::PathPCA::create() { return true; }


void megamol::thermodyn::PathPCA::release() {}


bool megamol::thermodyn::PathPCA::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        auto const pathStore = inCall->GetPathStore();
        auto const& entrySizes = inCall->GetEntrySize();
        auto inDirsPresent = inCall->HasDirections();
        auto inColsPresent = inCall->HasColors();

        par_data_.resize(pathStore->size());
        minV_.resize(pathStore->size());
        maxV_.resize(pathStore->size());

        pathStore_.resize(pathStore->size());
        dirsPresent_.resize(pathStore->size(), false);
        colsPresent_.resize(pathStore->size(), false);
        entrySizes_.resize(pathStore->size(), 3);

        for (size_t plidx = 0; plidx < pathStore->size(); ++plidx) {
            auto const& paths = pathStore->operator[](plidx);
            auto const entrySize = entrySizes[plidx];

            if (paths.empty()) continue;

            auto& data = par_data_[plidx];

            auto const rows = paths.size();
            auto const columns = inCall->GetTimeSteps();

            auto const dirPresent = inDirsPresent[plidx];
            dirsPresent_[plidx]=dirPresent;
            entrySizes_[plidx]=dirPresent?6:3;
            int dirOff = 3;
            if (inColsPresent[plidx]) {
                dirOff += 4;
            }

            data.resize(rows * columns * 4);

            Eigen::MatrixXf v1x(rows, columns);
            Eigen::MatrixXf v2x(rows, columns);
            Eigen::MatrixXf v1y(rows, columns);
            Eigen::MatrixXf v2y(rows, columns);
            Eigen::MatrixXf v1z(rows, columns);
            Eigen::MatrixXf v2z(rows, columns);

            size_t currow = 0;

            std::vector<float> vel(rows*columns*3);

            for (auto const& path : paths) {
                auto const& line = path.second;

                for (size_t fidx = 0; fidx < line.size(); fidx += entrySize) {
                    auto const curcol = fidx / entrySize;
                    auto const par_idx = currow + curcol * rows;
                    data[par_idx * 4 + 0] = line[fidx + 0];
                    data[par_idx * 4 + 1] = line[fidx + 1];
                    data[par_idx * 4 + 2] = line[fidx + 2];
                    if (dirPresent) {
                        vel[par_idx*3+0]=line[fidx+dirOff+0];
                        vel[par_idx*3+1]=line[fidx+dirOff+1];
                        vel[par_idx*3+2]=line[fidx+dirOff+2];
                    }


                    v1x(currow, curcol) = line[fidx + 0];
                    v1y(currow, curcol) = line[fidx + 1];
                    v1z(currow, curcol) = line[fidx + 2];


                    /*if (fidx > 0) {
                        v2x(currow, curcol - 1) = line[fidx + 0];
                        v2y(currow, curcol - 1) = line[fidx + 1];
                        v2z(currow, curcol - 1) = line[fidx + 2];
                    }*/
                }

                ++currow;
            }

            // svdsolver svd;
            // eigensolver es;

            // svd.compute(v1x, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // auto ux = svd.matrixU();
            // Eigen::MatrixXf sx = svd.singularValues().asDiagonal();
            // auto vx = svd.matrixV();

            // Eigen::MatrixXf stx = ux.transpose() * v2x * vx * sx.inverse();

            // vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Starting EV computation\n");

            // es.compute(stx);
            // vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Finished EV computation\n");
            // auto elx = es.eigenvalues();
            // auto evx = es.eigenvectors();

            // vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Starting EV recombination\n");

            //// Eigen::MatrixXf modes = ux*elx.real();
            // Eigen::MatrixXf modesX = ux * evx.real();
            // Eigen::MatrixXf modes = elx.real().inverse()*v2y*vx*sx.inverse()*evx.real();

            Eigen::MatrixXf modesX = computeModes(v1x);
            Eigen::MatrixXf modesY = computeModes(v1y);
            Eigen::MatrixXf modesZ = computeModes(v1z);

            vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Finished EV recombination\n");

            float minV = std::numeric_limits<float>::max();
            float maxV = std::numeric_limits<float>::lowest();

            float minX = std::numeric_limits<float>::max(), minY = std::numeric_limits<float>::max(),
                  minZ = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest(), maxY = std::numeric_limits<float>::lowest(),
                  maxZ = std::numeric_limits<float>::lowest();

            auto& paths_to_write = pathStore_[plidx];

            for (size_t ridx = 0; ridx < rows; ++ridx) {
                /*auto val = modes(ridx, 0);
                minV = std::min(minV, val);
                maxV = std::max(maxV, val);*/

                int ec = 3;
                std::vector<float> poss((columns) * ec);
                if (dirPresent) {
                    ec = 6;
                    poss.resize((columns) * ec);
                }

                for (size_t fidx = 0; fidx < columns; ++fidx) {
                    auto const idx = ridx + fidx * rows;
                    data[idx * 4 + 0] = modesX(ridx, fidx);
                    data[idx * 4 + 1] = modesY(ridx, fidx);
                    data[idx * 4 + 2] = modesZ(ridx, fidx);
                    data[idx * 4 + 3] = 0.0f;

                    poss[fidx * ec + 0] = modesX(ridx, fidx);
                    poss[fidx * ec + 1] = modesY(ridx, fidx);
                    poss[fidx * ec + 2] = modesZ(ridx, fidx);
                    if (dirPresent) {
                        poss[fidx * ec + 3] = vel[idx + 0];
                        poss[fidx * ec + 4] = vel[idx + 1];
                        poss[fidx * ec + 5] = vel[idx + 2];
                    }

                    // data[idx * 4 + 3] = val;
                    minX = std::min(minX, modesX(ridx, fidx));
                    minY = std::min(minY, modesY(ridx, fidx));
                    minZ = std::min(minZ, modesZ(ridx, fidx));
                    maxX = std::max(maxX, modesX(ridx, fidx));
                    maxY = std::max(maxY, modesY(ridx, fidx));
                    maxZ = std::max(maxZ, modesZ(ridx, fidx));
                }

                paths_to_write[ridx] = poss;
            }
            data.resize(rows * (columns) * 4);
            bbox_.Set(minX, minY, minZ, maxX, maxY, maxZ);

            minV_[plidx] = minV;
            maxV_[plidx] = maxV;

            vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Recoloring done min: %f max: %f\n", minV, maxV);
        }

        vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: BBOX - %f, %f, %f, %f, %f, %f\n", this->bbox_.GetLeft(),
            this->bbox_.GetBottom(), this->bbox_.GetBack(), this->bbox_.GetRight(), this->bbox_.GetTop(),
            this->bbox_.Front());
    }

    // par_data_.erase(std::remove_if(par_data_.begin(), par_data_.end(), [](auto const& el){return el.size() == 0;}),
    // par_data_.end());

    // vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Clear data\n");


    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);
    outCall->SetTimeSteps(inCall->GetTimeSteps());
    outCall->SetEntrySizes(entrySizes_);
    outCall->SetColorFlags(colsPresent_);
    outCall->SetDirFlags(dirsPresent_);
    outCall->SetPathStore(&pathStore_);

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox_);
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox_);
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    outCall->SetDataHash(inDataHash_);


    //outCall->SetParticleListCount(par_data_.size());
    //for (size_t plidx = 0; plidx < par_data_.size(); ++plidx) {
    //    if (par_data_[plidx].empty()) continue;
    //    auto& entry = outCall->AccessParticles(plidx);
    //    entry.SetCount(par_data_[plidx].size() / 4);
    //    entry.SetVertexData(
    //        core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, par_data_[plidx].data(), 4 * sizeof(float));
    //    /*entry.SetColourData(core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_I,
    //        par_data_[plidx].data() + 3, 4 * sizeof(float));*/
    //    entry.SetGlobalColour(255, 255, 255);
    //    entry.SetColourMapIndexValues(minV_[plidx], maxV_[plidx]);
    //    entry.SetGlobalRadius(this->bbox_.Width() / 1000.0f);
    //}

    // vislib::sys::Log::DefaultLog.WriteInfo("PathPCA: Output set\n");

    return true;
}


Eigen::MatrixXf megamol::thermodyn::PathPCA::computeModes(Eigen::MatrixXf const& x) const {
    svdsolver svd;
    eigensolver es;

    svd.compute(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto ux = svd.matrixU();
    Eigen::MatrixXf sx = svd.singularValues().asDiagonal();
    auto vx = svd.matrixV();

    Eigen::MatrixXf t = ux * sx;

    auto numFeatures = this->numFeatureSlot_.Param<core::param::IntParam>()->Value();

    numFeatures = numFeatures > t.cols() ? t.cols() : numFeatures;

    t.conservativeResize(t.rows(), numFeatures);
    vx.conservativeResize(vx.rows(), numFeatures);

    return t * vx.transpose();
}


bool megamol::thermodyn::PathPCA::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(1)) return false;


    /*outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());*/
    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox_);
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox_);
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);


    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);

    outCall->SetDataHash(inDataHash_);

    return true;
}
