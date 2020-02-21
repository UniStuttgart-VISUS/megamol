#include "stdafx.h"
#include "DepthFunction.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include <Eigen/LU>
#include "HD/HD.h"

using namespace megamol;
using namespace megamol::infovis;
using namespace Eigen;

enum DepthType { HALFSPACE_DEPTH = 0, FUNCTIONAL_DEPTH, MAHALANOBIS_DEPTH, SIMPLICAL_DEPTH };

Eigen::VectorXd DepthFunction::halfSpaceDepth(Eigen::MatrixXd dataMatrix) {
    Eigen::VectorXd result = Eigen::VectorXd(dataMatrix.rows());

    int nPoints = dataMatrix.rows();
    int nDims = dataMatrix.cols();

    double** x;
    x = new double*[nPoints];

    for (int i = 0; i < nPoints; i++) {
        x[i] = new double[nDims];
        for (int j = 0; j < nDims; j++) x[i][j] = dataMatrix(i, j);
    }

    double* z;
    z = new double[dataMatrix.cols()];

    for (int pIndex = 0; pIndex < nPoints; pIndex++) {
        for (int j = 0; j < dataMatrix.cols(); j++) {
            z[j] = dataMatrix(pIndex, j);
        }
        result(pIndex) = HalfspaceDepth::HD_Comb2(z, x, nPoints, nDims);
    }

    for (int k = 0; k < nPoints; k++) delete[] x[k];
    delete[] x;
    delete[] z;

    return result;
}

Eigen::VectorXd DepthFunction::functionalDepth(
    Eigen::MatrixXd dataMatrix, int samplesCount, int samplesLength, unsigned int seed) {
    srand(seed);

    Eigen::VectorXd result = Eigen::VectorXd(dataMatrix.rows());

    for (int row = 0; row < dataMatrix.rows(); row++) {
        Eigen::RowVectorXd dataPoint = dataMatrix.row(row);

        int hitsCount = 0;

        for (int sampleIndex = 0; sampleIndex < samplesCount; sampleIndex++) {
            Eigen::VectorXd rndIndices =
                (Eigen::VectorXd::Random(samplesLength).array() + 1.0) / 2.0 * (dataMatrix.rows() - 1.0);

            bool minMaxFound = true;

            for (int dim = 0; dim < dataMatrix.cols(); dim++) {

                double dataPointValue = dataPoint(dim);

                bool dim_maxFound = false;
                bool dim_minFound = false;

                for (int rndIndexCounter = 0; rndIndexCounter < samplesLength; rndIndexCounter++) {
                    int rndIndex = rndIndices(rndIndexCounter);

                    if (dataMatrix(rndIndex, dim) <= dataPointValue) dim_minFound = true;
                    if (dataMatrix(rndIndex, dim) >= dataPointValue) dim_maxFound = true;

                    if (dim_minFound && dim_maxFound) break;
                }

                // if a dimension was found where neither a min nor a max was found,
                // this datapoint isn't in the random set/sample
                if (!(dim_minFound && dim_maxFound)) {
                    minMaxFound = false;
                    break;
                }
            }

            if (minMaxFound) hitsCount++;
        }

        result(row) = (double)hitsCount / (double)samplesCount;
    }

    return result;
}

bool insideSimplexCheck(Eigen::VectorXd p, Eigen::MatrixXd x) {
    Eigen::MatrixXd tmp(x.rows() + 1, x.cols());
    tmp << x.transpose(), Eigen::MatrixXd::Ones(x.cols(), 1);
    tmp.transposeInPlace();
    double nominator = tmp.determinant();

    tmp.block(0, 0, x.rows(), 1) = p;

    double sum = 0;

    for (int pointIndex = 0; pointIndex < x.cols(); pointIndex++) {

        if (pointIndex != 0)
            tmp.block(0, pointIndex, x.rows(), 1) = x.block(0, pointIndex - 1, x.rows(), 1); // update matrix

        double sign = ((pointIndex % 2) * (-2.0) + 1.0);     // pow(-1,pointIndex)
        double alpha = sign * tmp.determinant() / nominator; // barycentric coordinate

        sum += alpha;
        if (alpha > 1.0 || alpha < 0.0) return false;
    }

    return true;
}

Eigen::VectorXd DepthFunction::simplicalDepth(Eigen::MatrixXd dataMatrix, int samplesCount, unsigned int seed) {
    srand(seed);

    int nPoints = dataMatrix.rows();
    int nDims = dataMatrix.cols();

    Eigen::VectorXd result = Eigen::VectorXd(dataMatrix.rows());

    for (int row = 0; row < dataMatrix.rows(); row++) {
        Eigen::VectorXd dataPoint = dataMatrix.row(row).transpose();

        int hitsCount = 0;

        for (int sampleIndex = 0; sampleIndex < samplesCount; sampleIndex++) {
            Eigen::VectorXd rndIndices =
                (Eigen::VectorXd::Random(nDims + 1).array() + 1.0) / 2.0 * (dataMatrix.rows() - 1.0);


            Eigen::MatrixXd vertices(nDims, nDims + 1);

            for (int rndIndexCounter = 0; rndIndexCounter < nDims + 1; rndIndexCounter++) {
                int rndIndex = rndIndices(rndIndexCounter);

                vertices.block(0, rndIndexCounter, nDims, 1) = dataMatrix.row(rndIndex).transpose();
            }


            if (insideSimplexCheck(dataPoint, vertices)) hitsCount++;
        }

        result(row) = (double)hitsCount / (double)samplesCount;
    }

    return result;
}

Eigen::MatrixXd DepthFunction::mahalanobisDepth(Eigen::MatrixXd dataMatrix) {
    int nPoints = dataMatrix.rows();

    Eigen::MatrixXd mahalaDepth = Eigen::MatrixXd::Zero(nPoints, 1);
    Eigen::MatrixXd invS =
        (1.0 / (nPoints - 1.0) * dataMatrix.transpose() *
            (Eigen::MatrixXd::Identity(nPoints, nPoints) - 1.0 / nPoints * Eigen::MatrixXd::Ones(nPoints, nPoints)) *
            dataMatrix)
            .inverse(); // inverse covariance Matrix

    Eigen::MatrixXd mu = 1.0 / nPoints * dataMatrix.transpose() * Eigen::MatrixXd::Ones(nPoints, 1);
    Eigen::MatrixXd xi;

    //#pragma omp parallel for //doesn't work
    for (int index = 0; index < nPoints; index++) {
        xi = dataMatrix.row(index).transpose();
        mahalaDepth(index, 0) = 1.0 / (1.0 + ((xi - mu).transpose() * invS * (xi - mu))(0, 0));
    }
    return mahalaDepth;
}

std::vector<std::string> split(std::string str, std::string token) {
    std::vector<std::string> result;
    while (str.size()) {
        int index = str.find(token);
        if (index != std::string::npos) {
            result.push_back(str.substr(0, index));
            str = str.substr(index + token.size());
            if (str.size() == 0) result.push_back(str);
        } else {
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

/// Parse string a with format columnIndex1;columnIndex2,columnIndex3,...;...
std::vector<std::vector<int>> parseColumnGroups(std::string& columnGroupString) {
    std::vector<std::vector<int>> columnGroups;
    auto groupStrings = split(columnGroupString, ";");
    for (auto groupString : groupStrings) {
        auto columnStrings = split(groupString, ",");
        std::vector<int> columns;
        for (auto columnString : columnStrings) {
            columns.push_back(std::stoi(columnString));
        }
        columnGroups.push_back(columns);
    }
    return columnGroups;
}

// ---------------------------------------------------------------------------

DepthFunction::DepthFunction(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Ouput")
    , dataInSlot("dataIn", "Input")
    , datahash(0)
    , dataInHash(0)
    , columnInfos()
    , columnGroupsSlot("columnGroups", "Semicolon-separated groups of comma-separated column indices to compute "
                                       "the data depth for. Defaults to one group, all columns.")
    , depthType("depthType", "The depth function to use for computing the depth statistics")
    , sampleCount("sampleCount", "The number of samples, that will be drawn (functional depth only)")
    , sampleLength("sampleLength", "The length of one sample (functional depth only)")
    , randomSeed("randomSeed", "The random seed (functional depth only)") {

    // Data input slot
    this->dataInSlot.SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    // Data output slot
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(0), &DepthFunction::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(1), &DepthFunction::getHashCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // Parameters
    columnGroupsSlot << new ::megamol::core::param::StringParam("");
    this->MakeSlotAvailable(&columnGroupsSlot);

    auto types = new ::megamol::core::param::EnumParam(1);
    types->SetTypePair(HALFSPACE_DEPTH, "Halfspace Depth");
    types->SetTypePair(FUNCTIONAL_DEPTH, "Functional Band Depth");
    types->SetTypePair(MAHALANOBIS_DEPTH, "Mahalanobis Depth");
    types->SetTypePair(SIMPLICAL_DEPTH, "Simplical Depth");
    depthType << types;
    this->MakeSlotAvailable(&depthType);

    sampleCount << new ::megamol::core::param::IntParam(20000);
    this->MakeSlotAvailable(&sampleCount);

    sampleLength << new ::megamol::core::param::IntParam(10);
    this->MakeSlotAvailable(&sampleLength);

    randomSeed << new ::megamol::core::param::IntParam(1337);
    this->MakeSlotAvailable(&randomSeed);

    // Add all parameters to common-parameter-handling list
    params.push_back(&columnGroupsSlot);
    params.push_back(&depthType);
    params.push_back(&sampleCount);
    params.push_back(&sampleLength);
    params.push_back(&randomSeed);
}

DepthFunction::~DepthFunction(void) { this->Release(); }

bool DepthFunction::create(void) { return true; }

void DepthFunction::release(void) {}

bool DepthFunction::getDataCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::table::TableDataCall* outCall =
            dynamic_cast<megamol::stdplugin::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID()); // inCall->Set? not outCall???
        if (!(*inCall)()) return false;

        bool finished = apply(inCall);
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

bool DepthFunction::getHashCallback(core::Call& c) {
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

void megamol::infovis::DepthFunction::assertData(megamol::stdplugin::datatools::table::TableDataCall* inCall) {
    auto columnCountIn = inCall->GetColumnsCount();
    auto columnCountOut = inCall->GetColumnsCount();
    auto rowsCount = inCall->GetRowsCount();
    auto inData = inCall->GetData();

    // Load data in a Matrix
    inDataMat = Eigen::MatrixXd(rowsCount, columnCountOut);
    for (int row = 0; row < rowsCount; row++) {
        for (int col = 0; col < columnCountIn; col++) {
            inDataMat(row, col) = inData[row * columnCountIn + col];
        }
    }
}

bool megamol::infovis::DepthFunction::paramsIsDirty() {
    for (auto param : params) {
        if (param->IsDirty()) return true;
    }

    return false;
}

void megamol::infovis::DepthFunction::paramsResetDirty() {
    for (auto param : params) {
        param->ResetDirty();
    }
}

bool megamol::infovis::DepthFunction::apply(megamol::stdplugin::datatools::table::TableDataCall* inCall) {
    // Test if input data or parameters have changed.
    if (this->dataInHash == inCall->DataHash() && !paramsIsDirty()) {
        // Do nothing since parameters are the same as before
        return true;
    }

    assertData(inCall);

    std::string columnGroupsString(this->columnGroupsSlot.Param<core::param::StringParam>()->Value().PeekBuffer());
    std::vector<std::vector<int>> columnGroups = parseColumnGroups(columnGroupsString);

    // Default to one group, containing all columns, if empty.
    if (columnGroups.empty()) {
        std::vector<int> columns;
        for (int i = 0; i < inDataMat.cols(); ++i) {
            columns.push_back(i);
        }
        columnGroups.push_back(columns);
    }

    // Compute depths for all column groups.
    Eigen::MatrixXd groupDepths = Eigen::MatrixXd::Zero(inDataMat.rows(), columnGroups.size());
    for (int group = 0; group < columnGroups.size(); group++) {
        auto columns = columnGroups[group];

        // Copy the columns together.
        Eigen::MatrixXd columnGroupMat = Eigen::MatrixXd::Zero(inDataMat.rows(), columns.size());
        for (int i = 0; i < columns.size(); i++) {
            auto column = columns[i];
            columnGroupMat.block(0, i, inDataMat.rows(), 1) = inDataMat.block(0, column, inDataMat.rows(), 1);
        }

        // Compute depth function.
        switch (this->depthType.Param<core::param::EnumParam>()->Value()) {
        case HALFSPACE_DEPTH:
            groupDepths.block(0, group, inDataMat.rows(), 1) = halfSpaceDepth(columnGroupMat);
            break;
        case FUNCTIONAL_DEPTH:
            groupDepths.block(0, group, inDataMat.rows(), 1) =
                functionalDepth(columnGroupMat, this->sampleCount.Param<core::param::IntParam>()->Value(),
                    this->sampleLength.Param<core::param::IntParam>()->Value(),
                    this->randomSeed.Param<core::param::IntParam>()->Value());
            break;
        case MAHALANOBIS_DEPTH:
            groupDepths.block(0, group, inDataMat.rows(), 1) = mahalanobisDepth(columnGroupMat);
            break;
        case SIMPLICAL_DEPTH:
            groupDepths.block(0, group, inDataMat.rows(), 1) =
                simplicalDepth(inDataMat, this->sampleCount.Param<core::param::IntParam>()->Value(),
                    this->randomSeed.Param<core::param::IntParam>()->Value());
            break;
        }
    }

    // Generate new column infos.
    this->columnInfos.clear();
    this->columnInfos.resize(groupDepths.cols());

    for (int group = 0; group < groupDepths.cols(); group++) {
        auto depths = groupDepths.col(group);
        std::string name = "Depth";
        for (int i = 0; i < columnGroups[group].size(); ++i) {
            name += " " + inCall->GetColumnsInfos()[columnGroups[group][i]].Name();
        }
        this->columnInfos[group]
            .SetName(name)
            .SetType(megamol::stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE)
            .SetMinimumValue(depths.minCoeff())
            .SetMaximumValue(depths.maxCoeff());
    }

    // Set matrix as output.
    this->data.clear();
    this->data.reserve(groupDepths.rows() * groupDepths.cols());
    for (size_t row = 0; row < groupDepths.rows(); row++) {
        for (int group = 0; group < groupDepths.cols(); group++) {
            this->data.push_back(groupDepths(row, group));
        }
    }

    // Update hash.
    this->dataInHash = inCall->DataHash();
    this->datahash++;

    // Reset parameters.
    paramsResetDirty();

    return true;
}
