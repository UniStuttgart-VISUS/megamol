/*
 * mmvtkmDataSource.cpp
 *
 * Copyright (C) 2020-2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */


#include "mmvtkm/mmvtkmDataSource.h"

#include <vtkm/Matrix.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>

#include "mmadios/CallADIOSData.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"


using namespace megamol;
using namespace megamol::mmvtkm;

typedef vtkm::Vec<float, 3> Vec3f;

/*
 * mmvtkmDataSource::mmvtkmDataSource
 */
mmvtkmDataSource::mmvtkmDataSource()
        : core::view::AnimDataModule()
        , getDataCalleeSlot_("getdata", "Slot to request data from this data source.")
        , nodesAdiosCallerSlot_("adiosNodeSlot", "Slot to request node data from adios.")
        , labelAdiosCallerSlot_("adiosLabelSlot", "Slot to request label data from adios.")
        , oldNodeDataHash_(0)
        , oldLabelDataHash_(0)
        , version_(0)
        , vtkmData_()
        , vtkmMetaData_()
        , vtkmDataFile_("") {
    this->getDataCalleeSlot_.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(0),
        &mmvtkmDataSource::getDataCallback); // GetData is FunctionName(0)
    this->getDataCalleeSlot_.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(1),
        &mmvtkmDataSource::getMetaDataCallback); // GetExtent is FunctionName(1)
    this->MakeSlotAvailable(&this->getDataCalleeSlot_);

    this->nodesAdiosCallerSlot_.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->nodesAdiosCallerSlot_);

    this->labelAdiosCallerSlot_.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->labelAdiosCallerSlot_);
}


/*
 * mmvtkmDataSource::~mmvtkmDataSource
 */
mmvtkmDataSource::~mmvtkmDataSource() {
    this->Release();
}


/*
 mmvtkmDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* mmvtkmDataSource::constructFrame() const {
    Frame* f = new Frame(*const_cast<mmvtkmDataSource*>(this));
    return f;
}


/*
 * mmvtkmDataSource::create
 */
bool mmvtkmDataSource::create() {
    return true;
}


/*
 * mmvtkmDataSource::loadFrame
 */
void mmvtkmDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {}


/*
 * mmvtkmDataSource::release
 */
void mmvtkmDataSource::release() {}


/*
 * mmvtkmDataSource::getDataCallback
 */
bool mmvtkmDataSource::getDataCallback(core::Call& caller) {
    mmvtkmDataCall* lhsVtkmDc = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (lhsVtkmDc == NULL)
        return false;

    adios::CallADIOSData* rhsTopNodesCad = this->nodesAdiosCallerSlot_.CallAs<adios::CallADIOSData>();
    if (rhsTopNodesCad == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError(
            "In %s at line %d. nodesCallADIOSData is nullptr.", __FILE__, __LINE__);
        return false;
    }

    adios::CallADIOSData* rhsBottomLabelCad = this->labelAdiosCallerSlot_.CallAs<adios::CallADIOSData>();
    if (rhsBottomLabelCad == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError(
            "In %s at line %d. labelCallADIOSData is nullptr.", __FILE__, __LINE__);
        return false;
    }

    // getDataCallback
    if (!(*rhsTopNodesCad)(0)) {
        core::utility::log::Log::DefaultLog.WriteError(
            "In %s at line %d. Error during nodes getData.", __FILE__, __LINE__);
        return false;
    }

    if (!(*rhsBottomLabelCad)(0)) {
        core::utility::log::Log::DefaultLog.WriteError(
            "In %s at line %d. Error during label getData.", __FILE__, __LINE__);
        return false;
    }


    size_t newNodeHash = rhsTopNodesCad->getDataHash(); // TODO how hash is set in tabletoadios or csvdatasource
    size_t newLabelHash = rhsBottomLabelCad->getDataHash();
    bool update = newNodeHash != oldNodeDataHash_ || newLabelHash != oldLabelDataHash_;

    if (update) {
        /////////////////////////////
        // get tetrahedron coordinates from csv file through adios
        /////////////////////////////
        try {
            std::string coordType = rhsTopNodesCad->getData("x")->getType();
            std::string labelType = rhsTopNodesCad->getData("node-label")->getType();

            // this assumes the data type from the file is float
            // TODO do for general case
            std::vector<float> nodeLabels = rhsTopNodesCad->getData("node-label")->GetAsFloat();
            std::vector<float> xCoord = rhsTopNodesCad->getData("x")->GetAsFloat();
            std::vector<float> yCoord = rhsTopNodesCad->getData("y")->GetAsFloat();
            std::vector<float> zCoord = rhsTopNodesCad->getData("z")->GetAsFloat();
            std::vector<float> s11 = rhsTopNodesCad->getData("s11")->GetAsFloat();
            std::vector<float> s12 = rhsTopNodesCad->getData("s12")->GetAsFloat();
            std::vector<float> s13 = rhsTopNodesCad->getData("s13")->GetAsFloat();
            std::vector<float> s21 = s12;
            std::vector<float> s22 = rhsTopNodesCad->getData("s22")->GetAsFloat();
            std::vector<float> s23 = rhsTopNodesCad->getData("s23")->GetAsFloat();
            std::vector<float> s31 = s13;
            std::vector<float> s32 = s23;
            std::vector<float> s33 = rhsTopNodesCad->getData("s33")->GetAsFloat();
            std::vector<float> hs1Value = rhsTopNodesCad->getData("hs1_wert")->GetAsFloat();
            std::vector<float> hs1X = rhsTopNodesCad->getData("hs1_x")->GetAsFloat();
            std::vector<float> hs1Y = rhsTopNodesCad->getData("hs1_y")->GetAsFloat();
            std::vector<float> hs1Z = rhsTopNodesCad->getData("hs1_z")->GetAsFloat();
            std::vector<float> hs2Value = rhsTopNodesCad->getData("hs2_wert")->GetAsFloat();
            std::vector<float> hs2X = rhsTopNodesCad->getData("hs2_x")->GetAsFloat();
            std::vector<float> hs2Y = rhsTopNodesCad->getData("hs2_y")->GetAsFloat();
            std::vector<float> hs2Z = rhsTopNodesCad->getData("hs2_z")->GetAsFloat();
            std::vector<float> hs3Value = rhsTopNodesCad->getData("hs3_wert")->GetAsFloat();
            std::vector<float> hs3X = rhsTopNodesCad->getData("hs3_x")->GetAsFloat();
            std::vector<float> hs3Y = rhsTopNodesCad->getData("hs3_y")->GetAsFloat();
            std::vector<float> hs3Z = rhsTopNodesCad->getData("hs3_z")->GetAsFloat();

            // TODO check if size of each vector is the same
            std::vector<float> elementLabel = rhsBottomLabelCad->getData("element-label")->GetAsFloat();
            std::vector<float> nodeA = rhsBottomLabelCad->getData("nodea")->GetAsFloat();
            std::vector<float> nodeB = rhsBottomLabelCad->getData("nodeb")->GetAsFloat();
            std::vector<float> nodeC = rhsBottomLabelCad->getData("nodec")->GetAsFloat();
            std::vector<float> nodeD = rhsBottomLabelCad->getData("noded")->GetAsFloat();


            int numElements = elementLabel.size();
            int numNodes = nodeLabels.size();
            vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
            vtkm::cont::DataSetFieldAdd dataSetFieldAdd;


            int numSkipped = 0;
            std::vector<Vec3f> points(numNodes);
            std::vector<Vec3f> pointHs1(numNodes);
            std::vector<Vec3f> pointHs2(numNodes);
            std::vector<Vec3f> pointHs3(numNodes);
            std::vector<Vec3f> pointHs(numNodes);
            std::vector<vtkm::Matrix<vtkm::Float32, 3, 3>> pointTensors(numNodes);
            //std::vector<vtkm::Id> cell_indices(numElements * 4);
            std::vector<vtkm::Id> cellIndices;
            std::vector<vtkm::IdComponent> numIndices;
            std::vector<vtkm::UInt8> cellShapes;


            for (int i = 0; i < numNodes; ++i) {
                Vec3f point = {xCoord[i], yCoord[i], zCoord[i]};
                Vec3f hs1 = {hs1X[i], hs1Y[i], hs1Z[i]};
                Vec3f hs2 = {hs2X[i], hs2Y[i], hs2Z[i]};
                Vec3f hs3 = {hs3X[i], hs3Y[i], hs3Z[i]};

                vtkm::Matrix<vtkm::Float32, 3, 3> tensor;
                tensor[0][0] = s11[i];
                tensor[0][1] = s12[i];
                tensor[0][2] = s13[i];
                tensor[1][0] = s21[i];
                tensor[1][1] = s22[i];
                tensor[1][2] = s23[i];
                tensor[2][0] = s31[i];
                tensor[2][1] = s32[i];
                tensor[2][2] = s33[i];

                Vec3f hs1TensProd = hs1Value[i] * vtkm::MatrixMultiply(tensor, hs1);
                Vec3f hs2TensProd = hs2Value[i] * vtkm::MatrixMultiply(tensor, hs2);
                Vec3f hs3TensProd = hs3Value[i] * vtkm::MatrixMultiply(tensor, hs3);

                points[i] = point;
                pointTensors[i] = tensor;
                pointHs1[i] = hs1;
                pointHs2[i] = hs2;
                pointHs3[i] = hs3;
                pointHs[i] = hs1TensProd + hs2TensProd + hs3TensProd;
            }


            for (int i = 0; i < numElements; ++i) {
                std::vector<int> labels = {(int)nodeA[i], (int)nodeB[i], (int)nodeC[i], (int)nodeD[i]};
                std::vector<vtkm::Id> indexBuffer(4);

                bool notFound = false;

                // BUILD TETRAHEDRON HERE
                // get vertex cooridnates for current tetrahedron vertices
                for (int j = 0; j < 4; ++j) {
                    auto it = std::find(nodeLabels.begin(), nodeLabels.end(), labels[j]);
                    int nodeIndex = std::distance(nodeLabels.begin(), it);
                    if (nodeIndex == nodeLabels.size()) {
                        // core::utility::log::Log::DefaultLog.WriteInfo("(%i, %i) with %i", i, j, labels[j]);
                        ++numSkipped;
                        notFound = true;
                        break;
                    }

                    indexBuffer[j] = nodeIndex;
                }

                if (notFound)
                    continue;

                cellShapes.emplace_back(vtkm::CELL_SHAPE_TETRA);
                numIndices.emplace_back(4); // tetrahedrons always have 4 vertices

                for (int j = 0; j < 4; ++j) {
                    cellIndices.emplace_back(indexBuffer[j]);
                }
            }

            core::utility::log::Log::DefaultLog.WriteInfo("Number of skipped tetrahedrons: %i", numSkipped);

            vtkmData_ = std::make_shared<VtkmData>();
            vtkmData_->data = dataSetBuilder.Create(points, cellShapes, numIndices, cellIndices);
            std::string field0 = "hs1";
            std::string field1 = "hs2";
            std::string field2 = "hs3";
            std::string field3 = "hs";
            dataSetFieldAdd.AddPointField(vtkmData_->data, field0, pointHs1);
            dataSetFieldAdd.AddPointField(vtkmData_->data, field1, pointHs2);
            dataSetFieldAdd.AddPointField(vtkmData_->data, field2, pointHs3);
            dataSetFieldAdd.AddPointField(vtkmData_->data, field3, pointHs);

            // vtkm::io::writer::VTKDataSetWriter writer("tetrahedron.vtk");
            // writer.WriteDataSet(vtkmData_);
            // core::utility::log::Log::DefaultLog.WriteInfo("vtkmData_ is successfully stored in tetrahedron.vtk.");

            // get min max bounds from dataset
            vtkmMetaData_.minMaxBounds = vtkmData_->data.GetCoordinateSystem(0).GetBounds();
            vtkmMetaData_.fieldNames = {field0, field1, field2, field3};

            lhsVtkmDc->setData(vtkmData_, ++version_);
            lhsVtkmDc->setMetaData(vtkmMetaData_);

            oldNodeDataHash_ = newNodeHash;
            oldLabelDataHash_ = newLabelHash;

            return true;
        } catch (const std::exception& e) {
            core::utility::log::Log::DefaultLog.WriteError("In %s at line %d. \n", __FILE__, __LINE__);
            core::utility::log::Log::DefaultLog.WriteError(e.what());

            return false;
        }
    }

    return true;
}


/*
 * mmvtkmDataSource::getMetaDataCallback
 */
bool mmvtkmDataSource::getMetaDataCallback(core::Call& caller) {
    mmvtkmDataCall* lhsVtkmDc = dynamic_cast<mmvtkmDataCall*>(&caller);

    adios::CallADIOSData* rhsTopNodesCad = this->nodesAdiosCallerSlot_.CallAs<adios::CallADIOSData>();
    if (rhsTopNodesCad == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("nodesCallADIOSData is nullptr");
        return false;
    }

    adios::CallADIOSData* rhsBottomLabelCad = this->labelAdiosCallerSlot_.CallAs<adios::CallADIOSData>();
    if (rhsBottomLabelCad == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("labelCallADIOSData is nullptr");
        return false;
    }

    lhsVtkmDc->setMetaData(vtkmMetaData_);

    return true;
}
