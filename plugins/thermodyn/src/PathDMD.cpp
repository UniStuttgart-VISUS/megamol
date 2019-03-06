#include "stdafx.h"
#include "PathDMD.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "thermodyn/PathLineDataCall.h"


megamol::thermodyn::PathDMD::PathDMD()
    : dataInSlot_("dataIn", "Input of particle pathlines"), dataOutSlot_("dataOut", "Output of DMD modes") {
    dataInSlot_.SetCompatibleCall<PathLineDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &PathDMD::getDataCallback);
    dataOutSlot_.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &PathDMD::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);
}


megamol::thermodyn::PathDMD::~PathDMD() { this->Release(); }


bool megamol::thermodyn::PathDMD::create() { return true; }


void megamol::thermodyn::PathDMD::release() {}


bool megamol::thermodyn::PathDMD::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<PathLineDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(0)) return false;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        auto const pathStore = inCall->GetPathStore();
        auto const& entrySizes = inCall->GetEntrySize();

        par_data_.resize(pathStore->size());
        minV_.resize(pathStore->size());
        maxV_.resize(pathStore->size());

        for (size_t plidx = 0; plidx < pathStore->size(); ++plidx) {
            auto const& paths = pathStore->operator[](plidx);
            auto const entrySize = entrySizes[plidx];

            if (paths.empty()) continue;

            auto& data = par_data_[plidx];

            auto const rows = paths.size();
            auto const columns = inCall->GetTimeSteps();

            data.resize(rows * columns * 4);

            Eigen::MatrixXf v1x(rows, columns - 1);
            Eigen::MatrixXf v2x(rows, columns - 1);
            Eigen::MatrixXf v1y(rows, columns - 1);
            Eigen::MatrixXf v2y(rows, columns - 1);
            Eigen::MatrixXf v1z(rows, columns - 1);
            Eigen::MatrixXf v2z(rows, columns - 1);

            size_t currow = 0;

            for (auto const& path : paths) {
                auto const& line = path.second;

                for (size_t fidx = 0; fidx < line.size(); fidx += entrySize) {
                    auto const curcol = fidx / entrySize;
                    auto const par_idx = currow + curcol * rows;
                    data[par_idx * 4 + 0] = line[fidx + 0];
                    data[par_idx * 4 + 1] = line[fidx + 1];
                    data[par_idx * 4 + 2] = line[fidx + 2];

                    if (curcol < columns - 1) {
                        v1x(currow, curcol) = line[fidx + 0];
                        v1y(currow, curcol) = line[fidx + 1];
                        v1z(currow, curcol) = line[fidx + 2];
                    }

                    if (fidx > 0) {
                        v2x(currow, curcol - 1) = line[fidx + 0];
                        v2y(currow, curcol - 1) = line[fidx + 1];
                        v2z(currow, curcol - 1) = line[fidx + 2];
                    }
                }

                ++currow;
            }

            //svdsolver svd;
            //eigensolver es;

            //svd.compute(v1x, Eigen::ComputeThinU | Eigen::ComputeThinV);
            //auto ux = svd.matrixU();
            //Eigen::MatrixXf sx = svd.singularValues().asDiagonal();
            //auto vx = svd.matrixV();

            //Eigen::MatrixXf stx = ux.transpose() * v2x * vx * sx.inverse();

            //vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Starting EV computation\n");

            //es.compute(stx);
            //vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Finished EV computation\n");
            //auto elx = es.eigenvalues();
            //auto evx = es.eigenvectors();

            //vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Starting EV recombination\n");

            //// Eigen::MatrixXf modes = ux*elx.real();
            //Eigen::MatrixXf modesX = ux * evx.real();
            // Eigen::MatrixXf modes = elx.real().inverse()*v2y*vx*sx.inverse()*evx.real();

            Eigen::MatrixXf modesX = computeModes(v1x, v2x);
            Eigen::MatrixXf modesY = computeModes(v1y, v2y);
            Eigen::MatrixXf modesZ = computeModes(v1z, v2z);

            vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Finished EV recombination\n");

            float minV = std::numeric_limits<float>::max();
            float maxV = std::numeric_limits<float>::lowest();

            float minX = std::numeric_limits<float>::max(), minY = std::numeric_limits<float>::max(), minZ = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest(), maxY = std::numeric_limits<float>::lowest(), maxZ = std::numeric_limits<float>::lowest();

            for (size_t ridx = 0; ridx < rows; ++ridx) {
                /*auto val = modes(ridx, 0);
                minV = std::min(minV, val);
                maxV = std::max(maxV, val);*/
                for (size_t fidx = 0; fidx < columns-1; ++fidx) {
                    auto const idx = ridx + fidx * rows;
                    data[idx * 4 + 0] = modesX(ridx, fidx);
                    data[idx * 4 + 1] = modesY(ridx, fidx);
                    data[idx * 4 + 2] = modesZ(ridx, fidx);
                    data[idx * 4 + 3] = 0.0f;
                    //data[idx * 4 + 3] = val;
                    minX = std::min(minX, modesX(ridx, fidx));
                    minY = std::min(minY, modesY(ridx, fidx));
                    minZ = std::min(minZ, modesZ(ridx, fidx));
                    maxX = std::max(maxX, modesX(ridx, fidx));
                    maxY = std::max(maxY, modesY(ridx, fidx));
                    maxZ = std::max(maxZ, modesZ(ridx, fidx));
                }
            }
            data.resize(rows*(columns-1)*4);
            bbox_.Set(minX, minY, minZ, maxX, maxY, maxZ);

            minV_[plidx] = minV;
            maxV_[plidx] = maxV;

            vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Recoloring done min: %f max: %f\n", minV, maxV);
        }
    }

    // par_data_.erase(std::remove_if(par_data_.begin(), par_data_.end(), [](auto const& el){return el.size() == 0;}),
    // par_data_.end());

    // vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Clear data\n");

    outCall->SetParticleListCount(par_data_.size());
    for (size_t plidx = 0; plidx < par_data_.size(); ++plidx) {
        if (par_data_[plidx].empty()) continue;
        auto& entry = outCall->AccessParticles(plidx);
        entry.SetCount(par_data_[plidx].size() / 4);
        entry.SetVertexData(
            core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, par_data_[plidx].data(), 4 * sizeof(float));
        /*entry.SetColourData(core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_I,
            par_data_[plidx].data() + 3, 4 * sizeof(float));*/
        entry.SetGlobalColour(255, 255, 255);
        entry.SetColourMapIndexValues(minV_[plidx], maxV_[plidx]);
        entry.SetGlobalRadius(this->bbox_.Width()/1000.0f);
    }

    // vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Output set\n");

    return true;
}


Eigen::MatrixXf megamol::thermodyn::PathDMD::computeModes(Eigen::MatrixXf const& v1, Eigen::MatrixXf const& v2) const {
    svdsolver svd;
    eigensolver es;

    svd.compute(v1, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto ux = svd.matrixU();
    Eigen::MatrixXf sx = svd.singularValues().asDiagonal();
    auto vx = svd.matrixV();

    Eigen::MatrixXf stx = ux.transpose() * v2 * vx * sx.inverse();

    vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Starting EV computation\n");

    es.compute(stx);
    vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Finished EV computation\n");
    auto elx = es.eigenvalues();
    auto evx = es.eigenvectors();

    vislib::sys::Log::DefaultLog.WriteInfo("PathDMD: Starting EV recombination\n");

    return ux * evx.imag();
}


bool megamol::thermodyn::PathDMD::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
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
