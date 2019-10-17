#include "stdafx.h"
#include "PCAProjection.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <set>
#include <sstream>
#include "MDSProjection.h"


using namespace megamol;
using namespace megamol::infovis;
using namespace Eigen;


MDSProjection::MDSProjection(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Ouput")
    , dataInSlot("dataIn", "Input")
    , reduceToNSlot("nComponents", "Number of components (dimensions) to keep")
    , datahash(0)
    , dataInHash(0)
    , columnInfos() {

    this->dataInSlot.SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(0), &MDSProjection::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(1), &MDSProjection::getHashCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    reduceToNSlot << new ::megamol::core::param::IntParam(2);
    this->MakeSlotAvailable(&reduceToNSlot);
}

MDSProjection::~MDSProjection(void) { this->Release(); }

bool MDSProjection::create(void) { return true; }

void MDSProjection::release(void) {}

bool MDSProjection::getDataCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::table::TableDataCall* outCall =
            dynamic_cast<megamol::stdplugin::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)()) return false;

        bool finished = dataProjection(inCall);
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

bool MDSProjection::getHashCallback(core::Call& c) {
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

bool megamol::infovis::MDSProjection::dataProjection(megamol::stdplugin::datatools::table::TableDataCall* inCall) {
    // Test if inData has changed and if slots have changed
    if (this->dataInHash == inCall->DataHash()) {
        if (!reduceToNSlot.IsDirty()) {
            return true; // Nothing to do
        }
    }

    auto columnCount = inCall->GetColumnsCount();
    auto column_infos = inCall->GetColumnsInfos();
    auto rowsCount = inCall->GetRowsCount();
    auto inData = inCall->GetData();

    int outputDimCount = this->reduceToNSlot.Param<core::param::IntParam>()->Value();
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

    // generate dissimilarity Matrix( squared euclidean Distance matrix)
    Eigen::MatrixXd delta2 = euclideanDissimilarityMatrix(inDataMat).array().pow(2);
    // compute MDS
    Eigen::MatrixXd result = classicMds(delta2, outputDimCount);

    // generate new columns
    this->columnInfos.clear();
    this->columnInfos.resize(outputDimCount);

    for (int indexX = 0; indexX < outputDimCount; indexX++) {
        columnInfos[indexX]
            .SetName("MDS" + std::to_string(indexX))
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

    return true;
}

Eigen::MatrixXd megamol::infovis::MDSProjection::euclideanDissimilarityMatrix(Eigen::MatrixXd dataMatrix) {
    // generate euclidean Distance matrix
    int rowsCount = dataMatrix.rows();
    Eigen::MatrixXd distanceMatrix = Eigen::MatrixXd::Zero(rowsCount, rowsCount);
    for (int row = 1; row < rowsCount; row++) {
        for (int col = 0; col < row; col++) {
            double distance = (dataMatrix.row(row) - dataMatrix.row(col)).norm();

            distanceMatrix(row, col) = distance;
            distanceMatrix(col, row) = distance;
        }
    }

    return distanceMatrix;
}

Eigen::MatrixXd megamol::infovis::MDSProjection::classicMds(
    Eigen::MatrixXd squaredDissimilarityMatrix, int outputDimension) {
    // centering matrix

    int rowsCount = squaredDissimilarityMatrix.rows();
    assert(squaredDissimilarityMatrix.rows() == squaredDissimilarityMatrix.cols());

    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(rowsCount, rowsCount) -
                        (1.0 / (double)rowsCount) * Eigen::MatrixXd::Ones(rowsCount, rowsCount);

    // Apply double centering
    Eigen::MatrixXd B = -0.5 * J * squaredDissimilarityMatrix * J;

    // Compute eigenvalues and eigenvectors
    EigenSolver<MatrixXd> eigSolver(B);

    VectorXd eigVal = eigSolver.eigenvalues().real();
    MatrixXd eigVec = eigSolver.eigenvectors().real();

    // Sort eigenvalues (with index): descending
    // Each eigenvalue represents variance
    typedef std::pair<float, int> eigenPair;
    std::vector<eigenPair> sorted;
    for (unsigned int i = 0; i < eigVec.cols(); ++i) {
        sorted.push_back(std::make_pair(eigVal(i), i));
    }
    std::sort(sorted.begin(), sorted.end(), [&sorted](eigenPair& a, eigenPair& b) { return a.first > b.first; });

    // Create Matrix out of sorted (and selected) eigenvectors, with the frobenius norm
    // of an eigenvector beeing the corresponding eigenvalue. lambda = |v|^2
    MatrixXd result = MatrixXd(eigVec.rows(), outputDimension);
    for (int i = 0; i < outputDimension; ++i) {
        result.col(i) = eigVec.col(sorted[i].second) * sqrt(abs(eigVal(sorted[i].second)));
    }

    return result;
}

Eigen::MatrixXd megamol::infovis::MDSProjection::bMatrix(
    Eigen::MatrixXd X, Eigen::MatrixXd W, Eigen::MatrixXd dissimilarityMatrix) {
    assert(X.rows() == W.rows());
    assert(X.rows() == dissimilarityMatrix.rows());
    assert(W.rows() == W.cols());
    assert(dissimilarityMatrix.rows() == dissimilarityMatrix.cols());
    int nPoints = X.rows();

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nPoints, nPoints);

    // upper and lower triangle Matrix
    for (int i = 1; i < nPoints; i++) {
        for (int j = 0; j < i; j++) {
            double distance = (X.row(i) - X.row(j)).norm();
            if (distance == 0.0) {
                B(i, j) = 0;
                B(j, i) = 0;
            } else {
                double Bij = -W(i, j) * dissimilarityMatrix(i, j) / distance;
                B(i, j) = Bij;
                B(j, i) = Bij;
            }
        }
    }
    // diagonal
    for (int i = 0; i < nPoints; i++) {
        B(i, i) = -B.row(i).sum();
    }

    return B;
}

Eigen::MatrixXd megamol::infovis::MDSProjection::vMatrix(Eigen::MatrixXd W) {
    assert(W.rows() == W.cols());
    int nPoints = W.rows();

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nPoints, nPoints);

    // upper and lower triangle Matrix
    for (int i = 1; i < nPoints; i++) {
        for (int j = 0; j < i; j++) {
            B(i, j) = -W(i, j);
            B(j, i) = -W(i, j);
        }
    }

    // diagonal
    for (int i = 0; i < nPoints; i++) {
        B(i, i) = -W.row(i).sum();
    }

    return B;
}

Eigen::MatrixXd megamol::infovis::MDSProjection::smacofMds(Eigen::MatrixXd dissimilarityMatrix, int outputDimension,
    int countSteps, Eigen::MatrixXd weightsMatrix, double tolerance) {
    assert(dissimilarityMatrix.rows() == dissimilarityMatrix.cols());

    // centering matrix
    int nPoints = dissimilarityMatrix.rows();
    bool weightsAllOne = weightsMatrix(0, 0) == 1 && weightsMatrix.cols() == 1 && weightsMatrix.rows() == 1;
    Eigen::MatrixXd Vp;
    if (!weightsAllOne) {
        Vp = (vMatrix(weightsMatrix) + Eigen::MatrixXd::Ones(nPoints, nPoints)).inverse() -
             pow(nPoints, -2) * Eigen::MatrixXd::Ones(nPoints, nPoints);
    } else {
        weightsMatrix = Eigen::MatrixXd::Ones(nPoints, nPoints);
    }

    // start algortihm
    // initial X
    const bool randomInit = true; // TODO: param
    Eigen::MatrixXd X;
    if (randomInit) {
        srand(1337); // TODO: param
        X = 100.0 * Eigen::MatrixXd::Random(dissimilarityMatrix.rows(), outputDimension);
    } else {
        // Initialize with classic MDS or something else good
        X = classicMds(dissimilarityMatrix.array().square(), outputDimension);
    }

    // stress
    double curStress = stress(dissimilarityMatrix, X, weightsMatrix);
    double oldStress;
    int k;

    for (k = 0; k < countSteps; k++) {
        if (weightsAllOne) {
            Eigen::MatrixXd B = bMatrix(X, weightsMatrix, dissimilarityMatrix);
            X = ((B * X).array()) / (float)nPoints;

        } else {
            X = Vp * bMatrix(X, weightsMatrix, dissimilarityMatrix) * X;
        }
        oldStress = curStress;
        curStress = stress(dissimilarityMatrix, X, weightsMatrix);

        // Early termination
        if (abs(oldStress - curStress) < tolerance) {
            break;
        }
    }

    // vislib::sys::Log::DefaultLog.WriteInfo("MDS Computation:\n"
    //                                       " Min/Max Weight: %f / %f\n"
    //                                       " Steps: %i\n"
    //                                       " Stress (Delta): %f ( %f )\n",
    //    weightsMatrix.maxCoeff(), weightsMatrix.minCoeff(), k, curStress, oldStress - curStress);

    return X;
}

Eigen::MatrixXd megamol::infovis::MDSProjection::ordinalMds(Eigen::MatrixXd dissimilarityMatrix, int outputDimension,
    int countSteps, Eigen::MatrixXd weightsMatrix, double tolerance) {

    // start algortihm
    // initial X
    bool randomInit = true;
    Eigen::MatrixXd X;
    if (randomInit) {
        srand(1337);                                                                      // same Seed
        X = 100.0 * Eigen::MatrixXd::Random(dissimilarityMatrix.cols(), outputDimension); // random init with the same
    } else {
        X = classicMds(dissimilarityMatrix.array().square(), outputDimension); // or random //or something else good
    }

    // sort each row and generate an Index Matrix

    Eigen::MatrixXd sortedIndexes = Eigen::MatrixXd::Zero(dissimilarityMatrix.rows(), dissimilarityMatrix.cols());

    // row schleife
    for (int row = 0; row < dissimilarityMatrix.rows(); row++) {
        typedef std::pair<float, int> indexValuePair;
        std::vector<indexValuePair> sorted;
        for (unsigned int i = 0; i < dissimilarityMatrix.cols(); ++i) {
            sorted.push_back(std::make_pair(dissimilarityMatrix(row, i), i)); //<-----bei dissmatrix richtige Indexe
        }
        std::sort(sorted.begin(), sorted.end(),
            [&sorted](indexValuePair& a, indexValuePair& b) { return a.first < b.first; });
        for (unsigned int i = 0; i < dissimilarityMatrix.cols(); ++i) {
            sortedIndexes(row, i) = sorted[i].second;
        }
    }

    // start: modified Kruskal-Shepard
    for (int i = 0; i < countSteps; i++) {
        // calculate distances
        Eigen::MatrixXd distances = euclideanDissimilarityMatrix(X);
        Eigen::MatrixXd newDistances = distances;

        for (int row = 0; row < dissimilarityMatrix.rows(); row++) {

            int medianIndex;
            dissimilarityMatrix.row(row).minCoeff(&medianIndex);

            std::set<int> notMonotoncols;

            bool consecutive = false;

            for (int sortedIndex = 0; sortedIndex < dissimilarityMatrix.cols() - 1; sortedIndex++) {

                int dissA = sortedIndexes(row, sortedIndex);
                int dissB = sortedIndexes(row, sortedIndex + 1);

                if (dissA == medianIndex) continue;
                if (dissB == medianIndex) continue;

                bool valueAdded = false;

                if ((dissimilarityMatrix(row, dissA) < dissimilarityMatrix(row, dissB)) &&
                    (distances(row, dissA) > distances(row, dissB))) {
                    valueAdded = true;
                }
                if ((dissimilarityMatrix(row, dissA) > dissimilarityMatrix(row, dissB)) &&
                    (distances(row, dissA) < distances(row, dissB))) {
                    valueAdded = true;
                }
                if ((dissimilarityMatrix(row, dissA) == dissimilarityMatrix(row, dissB)) &&
                    (distances(row, dissA) != distances(row, dissB))) {
                    // valueAdded = true;
                }

                if (valueAdded) {
                    notMonotoncols.insert(dissA);
                    notMonotoncols.insert(dissB);

                    consecutive = true;
                }

                if ((valueAdded == false) && (consecutive == true)) {
                    consecutive = false;

                    if (notMonotoncols.size() != 0) {
                        double avg = 0;
                        for (int i : notMonotoncols) {
                            avg += distances(row, i);
                        }
                        avg /= (double)notMonotoncols.size();

                        for (int i : notMonotoncols) newDistances(row, i) = avg;

                        notMonotoncols.clear();
                    }
                }
            }
        }
        Eigen::MatrixXd Xold = Eigen::MatrixXd::Zero(dissimilarityMatrix.cols(), outputDimension);
        double alpha = 1;

        Xold = X;

        for (size_t i = 0, nRows = distances.rows(), nCols = distances.cols(); i < nRows; ++i)
            for (size_t j = 0; j < nRows; j++) {
                if (i == j) continue;
                X.row(i) = X.row(i).array() + alpha / (dissimilarityMatrix.cols() - 1) *
                                                  (1 - newDistances(i, j) / distances(i, j)) *
                                                  (Xold.row(j).array() - Xold.row(i).array());
            }
    }

    return X;
}

double megamol::infovis::MDSProjection::stress(
    Eigen::MatrixXd dissimilarityMatrix, Eigen::MatrixXd dataPointsMatrix, Eigen::MatrixXd weightsMatrix) {

    assert(dissimilarityMatrix.rows() == dissimilarityMatrix.cols());
    int nPoints = dissimilarityMatrix.rows();
    assert(dissimilarityMatrix.rows() == dataPointsMatrix.rows());
    int dims = dataPointsMatrix.cols();

    if (weightsMatrix(0, 0) == 1 && weightsMatrix.cols() == 1 && weightsMatrix.rows() == 1) {
        weightsMatrix = Eigen::MatrixXd::Ones(nPoints, nPoints);
    }

    double sum = 0;
    for (int j = 1; j < nPoints; j++) {
        for (int i = 0; i < j; i++) {
            double distance = (dataPointsMatrix.row(i) - dataPointsMatrix.row(j)).norm();
            sum += weightsMatrix(i, j) * pow(dissimilarityMatrix(i, j) - distance, 2);
        }
    }

    return sum;
}
