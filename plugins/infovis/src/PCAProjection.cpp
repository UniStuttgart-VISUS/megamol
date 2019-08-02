#include "stdafx.h"
#include "PCAProjection.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <sstream>


using namespace megamol;
using namespace megamol::infovis;
using namespace Eigen;


PCAProjection::PCAProjection(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Ouput")
    , dataInSlot("dataIn", "Input")
    , reduceToNSlot("nComponents", "Number of components (dimensions) to keep")
    , scaleSlot("scale", "Set to scale each column to unit variance")
    , centerSlot("center", "Set to shift the mean centroid to the origin")
    , datahash(0)
    , dataInHash(0)
    , columnInfos() {

    this->dataInSlot.SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(0), &PCAProjection::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(1), &PCAProjection::getHashCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    reduceToNSlot << new ::megamol::core::param::IntParam(2);
    this->MakeSlotAvailable(&reduceToNSlot);

    centerSlot << new ::megamol::core::param::BoolParam(true);
    this->MakeSlotAvailable(&centerSlot);

    scaleSlot << new ::megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&scaleSlot);
}


PCAProjection::~PCAProjection(void) { this->Release(); }

bool PCAProjection::create(void) { return true; }

void PCAProjection::release(void) {}

bool PCAProjection::getDataCallback(core::Call& c) {

    try {
        megamol::stdplugin::datatools::table::TableDataCall* outCall =
            dynamic_cast<megamol::stdplugin::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)()) return false;

        bool finished = project(inCall);
        if (finished == false) return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->datahash);

        // set outCall
        if (this->columnInfos.size() != 0) {
            outCall->Set(this->columnInfos.size(), this->data.size() / this->columnInfos.size(),
                this->columnInfos.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }

    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::processData\n"), ClassName());
        return false;
    }

    return true;
}

bool PCAProjection::getHashCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::table::TableDataCall* outCall =
            dynamic_cast<megamol::stdplugin::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1)) return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->datahash);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::getHashCallback\n"), ClassName());
        return false;
    }

    return true;
}

bool megamol::infovis::PCAProjection::project(megamol::stdplugin::datatools::table::TableDataCall* inCall) {

    // check if inData has changed and if Slots have changed
    if (this->dataInHash == inCall->DataHash()) {
        if (!reduceToNSlot.IsDirty() && !scaleSlot.IsDirty() && !centerSlot.IsDirty()) {
            return true; // Nothing to do
        }
    }


    auto columnCount = inCall->GetColumnsCount();
    auto column_infos = inCall->GetColumnsInfos();
    auto rowsCount = inCall->GetRowsCount();
    auto inData = inCall->GetData();

    unsigned int outputDimCount = this->reduceToNSlot.Param<core::param::IntParam>()->Value();
    bool center = this->centerSlot.Param<core::param::BoolParam>()->Value();
    bool scale = this->scaleSlot.Param<core::param::BoolParam>()->Value();


    if (outputDimCount <= 0 || outputDimCount > columnCount) {
        vislib::sys::Log::DefaultLog.WriteError(_T("%hs: No valid Dimension Count has been given\n"), ClassName());
        return false;
    }

    // Load data in a Matrix
    Eigen::MatrixXd inDataMat = Eigen::MatrixXd(rowsCount, columnCount);
    for (int row = 0; row < rowsCount; row++) {
        for (int col = 0; col < columnCount; col++) {
            inDataMat(row, col) = inData[row * columnCount + col];
        }
    }

    // calculate mean for each column
    Eigen::VectorXd mean_vector(columnCount);
    mean_vector = inDataMat.colwise().mean();


    // prepare data
    if (center) {
        // substract mean columnwise
        for (int col = 0; col < columnCount; col++) {
            inDataMat.col(col) -= Eigen::VectorXd::Constant(rowsCount, mean_vector(col));
        }
    }


    if (scale) {
        // scale data to unit variance by dividing by standard deviation
        Eigen::VectorXd stdDev(columnCount);
        for (int col = 0; col < columnCount; col++) {
            stdDev(col) = sqrt(inDataMat.col(col).cwiseProduct(inDataMat.col(col)).sum() / (rowsCount - 1));
            inDataMat.col(col) /= stdDev(col);
        }
    }


    // calculate CovarianceMatrix
    MatrixXd covarianceMatrix = inDataMat;

    /** if center is off: "R ggfortify" doesn't substract mean for the covariance matrix
    //substract mean for cov Matrix
    mean_vector = inDataMat.colwise().mean();
    for (int col = 0; col < columnCount; col++) {
        covarianceMatrix.col(col) -= Eigen::VectorXd::Constant(rowsCount, mean_vector(col));
    }*/


    covarianceMatrix = covarianceMatrix.transpose() * covarianceMatrix;
    covarianceMatrix = covarianceMatrix / (float)(rowsCount - 1);


    // calculate Eigenvalues and Eigenvectors
    EigenSolver<MatrixXd> eigSolver(covarianceMatrix);

    VectorXd eigVal = eigSolver.eigenvalues().real();
    MatrixXd eigVec = eigSolver.eigenvectors().real();

    // sort eigenvalues (with index): descending
    // each eigenvalue represents the variance
    typedef std::pair<float, int> eigenPair;
    std::vector<eigenPair> sorted;
    for (unsigned int i = 0; i < columnCount; ++i) {
        sorted.push_back(std::make_pair(eigVal(i), i));
    }
    std::sort(sorted.begin(), sorted.end(), [&sorted](eigenPair& a, eigenPair& b) { return a.first > b.first; });

    // create Matrix out of sorted (and selected) eigenvectors
    MatrixXd eigVecBasis = MatrixXd(columnCount, outputDimCount);
    for (unsigned int i = 0; i < outputDimCount; ++i) {
        eigVecBasis.col(i) = eigVec.col(sorted[i].second);
    }


    // calculate PCA
    MatrixXd result = inDataMat * eigVecBasis;


    //// center
    // mean_vector = result.colwise().mean();
    //// substract mean adjusted Matrix (shift to mean to zero)
    // for (int col = 0; col < result.cols(); col++) {
    //    result.col(col) -= Eigen::VectorXd::Constant(rowsCount, mean_vector(col));
    //}


    std::stringstream debug;
    debug << std::endl << result << std::endl;

    vislib::sys::Log::DefaultLog.WriteInfo(debug.str().c_str());

    // generate new columns
    this->columnInfos.clear();
    this->columnInfos.resize(outputDimCount);

    for (int indexX = 0; indexX < outputDimCount; indexX++) {
        columnInfos[indexX]
            .SetName("PC" + std::to_string(indexX))
            .SetType(megamol::stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE)
            .SetMinimumValue(result.col(indexX).minCoeff())
            .SetMaximumValue(result.col(indexX).maxCoeff());
    }

    // Result Matrix into Output
    this->data.clear();
    this->data.reserve(rowsCount * outputDimCount);

    for (size_t row = 0; row < rowsCount; row++) {
        for (size_t col = 0; col < outputDimCount; col++) this->data.push_back(result(row, col));
    }


    this->dataInHash = inCall->DataHash();
    this->datahash++;
    reduceToNSlot.ResetDirty();
    scaleSlot.ResetDirty();
    centerSlot.ResetDirty();

    return true;
}
