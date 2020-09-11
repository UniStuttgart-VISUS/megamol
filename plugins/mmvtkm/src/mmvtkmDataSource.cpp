/*
 * mmvtkmDataSource.cpp
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmvtkm/mmvtkmDataSource.h"

#include "adios_plugin/CallADIOSData.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "vtkm/cont/DataSetBuilderExplicit.h"
#include "vtkm/cont/DataSetFieldAdd.h"
#include "vtkm/io/writer/VTKDataSetWriter.h"


using namespace megamol;
using namespace megamol::mmvtkm;

typedef vtkm::Vec<float, 3> Vec3f;

/*
 * mmvtkmDataSource::mmvtkmDataSource
 */
mmvtkmDataSource::mmvtkmDataSource(void)
    : core::view::AnimDataModule()
    , getDataCalleeSlot_("getdata", "Slot to request data from this data source.")
    , nodesAdiosCallerSlot_("adiosNodeSlot", "Slot to request node data from adios.")
    , labelAdiosCallerSlot_("adiosLabelSlot", "Slot to request label data from adios.")
    , oldNodeDataHash_(0) 
	, oldLabelDataHash_(0)
    , vtkmData_()
    , vtkmDataFile_("")
    , minMaxBounds_(0.f, 0.f, 0.f, 1.f, 1.f, 1.f) 
{
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
mmvtkmDataSource::~mmvtkmDataSource(void) { this->Release(); }


/*
 mmvtkmDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* mmvtkmDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<mmvtkmDataSource*>(this));
    return f;
}


/*
 * mmvtkmDataSource::create
 */
bool mmvtkmDataSource::create(void) { return true; }


/*
 * mmvtkmDataSource::loadFrame
 */
void mmvtkmDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
}


/*
 * mmvtkmDataSource::release
 */
void mmvtkmDataSource::release(void) {
}


/*
 * mmvtkmDataSource::getDataCallback
 */
bool mmvtkmDataSource::getDataCallback(core::Call& caller) {
    mmvtkmDataCall* lhsVtkmDc = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (lhsVtkmDc == NULL) return false;

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

	size_t newNodeHash = rhsTopNodesCad->getDataHash();		// TODO how hash is set in tabletoadios or csvdatasource
	size_t newLabelHash = rhsBottomLabelCad->getDataHash();
    bool update = newNodeHash != oldNodeDataHash_ || newLabelHash != oldLabelDataHash_;


    if (update) {
		/////////////////////////////
		// get tetrahedron coordinates from csv file through adios
		/////////////////////////////
		std::string coord_type = rhsTopNodesCad->getData("x")->getType();
		std::string label_type = rhsTopNodesCad->getData("node-label")->getType();

		// this assumes the data type from the file is float
		// TODO do for general case
		std::vector<float> node_labels = rhsTopNodesCad->getData("node-label")->GetAsFloat();
		std::vector<float> x_coord = rhsTopNodesCad->getData("x")->GetAsFloat();
		std::vector<float> y_coord = rhsTopNodesCad->getData("y")->GetAsFloat();
		std::vector<float> z_coord = rhsTopNodesCad->getData("z")->GetAsFloat();
		std::vector<float> s11 = rhsTopNodesCad->getData("s11")->GetAsFloat();
		std::vector<float> s12 = rhsTopNodesCad->getData("s12")->GetAsFloat();
		std::vector<float> s13 = rhsTopNodesCad->getData("s13")->GetAsFloat();
		std::vector<float> s21 = s12;
		std::vector<float> s22 = rhsTopNodesCad->getData("s22")->GetAsFloat();
		std::vector<float> s23 = rhsTopNodesCad->getData("s23")->GetAsFloat();
		std::vector<float> s31 = s13;
		std::vector<float> s32 = s23;
		std::vector<float> s33 = rhsTopNodesCad->getData("s33")->GetAsFloat();
		std::vector<float> hs1_wert = rhsTopNodesCad->getData("hs1_wert")->GetAsFloat();
		std::vector<float> hs1_x = rhsTopNodesCad->getData("hs1_x")->GetAsFloat();
		std::vector<float> hs1_y = rhsTopNodesCad->getData("hs1_y")->GetAsFloat();
		std::vector<float> hs1_z = rhsTopNodesCad->getData("hs1_z")->GetAsFloat();
		std::vector<float> hs2_wert = rhsTopNodesCad->getData("hs2_wert")->GetAsFloat();
		std::vector<float> hs2_x = rhsTopNodesCad->getData("hs2_x")->GetAsFloat();
		std::vector<float> hs2_y = rhsTopNodesCad->getData("hs2_y")->GetAsFloat();
		std::vector<float> hs2_z = rhsTopNodesCad->getData("hs2_z")->GetAsFloat();
		std::vector<float> hs3_wert = rhsTopNodesCad->getData("hs3_wert")->GetAsFloat();
		std::vector<float> hs3_x = rhsTopNodesCad->getData("hs3_x")->GetAsFloat();
		std::vector<float> hs3_y = rhsTopNodesCad->getData("hs3_y")->GetAsFloat();
		std::vector<float> hs3_z = rhsTopNodesCad->getData("hs3_z")->GetAsFloat();

		// TODO check if size of each vector is the same (assert)
		std::vector<float> element_label = rhsBottomLabelCad->getData("element-label")->GetAsFloat();
		std::vector<float> nodeA = rhsBottomLabelCad->getData("nodea")->GetAsFloat();
		std::vector<float> nodeB = rhsBottomLabelCad->getData("nodeb")->GetAsFloat();
		std::vector<float> nodeC = rhsBottomLabelCad->getData("nodec")->GetAsFloat();
		std::vector<float> nodeD = rhsBottomLabelCad->getData("noded")->GetAsFloat();


		int num_elements = element_label.size();
		int num_nodes = node_labels.size();
		vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
		vtkm::cont::DataSetFieldAdd dataSetFieldAdd;


		int num_skipped = 0;
		std::vector<Vec3f> points(num_nodes);
		std::vector<Vec3f> point_hs1(num_nodes);
		std::vector<Vec3f> point_hs2(num_nodes);
		std::vector<Vec3f> point_hs3(num_nodes);
		std::vector<Vec3f> point_hs(num_nodes);
		std::vector<vtkm::Matrix<vtkm::Float32, 3, 3>> point_tensors(num_nodes);
		//std::vector<vtkm::Id> cell_indices(num_elements * 4);
		std::vector<vtkm::Id> cell_indices;
		std::vector<vtkm::IdComponent> num_indices;
		std::vector<vtkm::UInt8> cell_shapes;


		for (int i = 0; i < num_nodes; ++i) {
		    Vec3f point = {x_coord[i], y_coord[i], z_coord[i]};
		    Vec3f hs1 = {hs1_x[i], hs1_y[i], hs1_z[i]};
		    Vec3f hs2 = {hs2_x[i], hs2_y[i], hs2_z[i]};
		    Vec3f hs3 = {hs3_x[i], hs3_y[i], hs3_z[i]};

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

			Vec3f hs1_tens_prod = hs1_wert[i] * vtkm::MatrixMultiply(tensor, hs1);
		    Vec3f hs2_tens_prod = hs2_wert[i] * vtkm::MatrixMultiply(tensor, hs2);
		    Vec3f hs3_tens_prod = hs3_wert[i] * vtkm::MatrixMultiply(tensor, hs3);

			points[i] = point;
		    point_tensors[i] = tensor;
		    point_hs1[i] = hs1;
		    point_hs2[i] = hs2;
		    point_hs3[i] = hs3;
		    point_hs[i] = hs1_tens_prod + hs2_tens_prod + hs3_tens_prod;
		}


		for (int i = 0; i < num_elements; ++i) {
		    std::vector<int> labels = {(int)nodeA[i], (int)nodeB[i], (int)nodeC[i], (int)nodeD[i]};
		    std::vector<vtkm::Id> index_buffer(4);

		    bool not_found = false;

		    // BUILD TETRAHEDRON HERE
		    // get vertex cooridnates for current tetrahedron vertices
		    for (int j = 0; j < 4; ++j) {
		        auto it = std::find(node_labels.begin(), node_labels.end(), labels[j]);
		        int node_index = std::distance(node_labels.begin(), it);
		        if (node_index == node_labels.size()) {
		            // core::utility::log::Log::DefaultLog.WriteInfo("(%i, %i) with %i", i, j, labels[j]);
		            ++num_skipped;
		            not_found = true;
		            break;
		        }

		        index_buffer[j] = node_index;
		    }

		    if (not_found) continue;

			cell_shapes.emplace_back(vtkm::CELL_SHAPE_TETRA);
		    num_indices.emplace_back(4);	// tetrahedrons always have 4 vertices

		    for (int j = 0; j < 4; ++j) {
		        cell_indices.emplace_back(index_buffer[j]);
		    }
		}

		core::utility::log::Log::DefaultLog.WriteInfo("Number of skipped tetrahedrons: %i", num_skipped);

		vtkmData_ = dataSetBuilder.Create(points, cell_shapes, num_indices, cell_indices);
		dataSetFieldAdd.AddPointField(vtkmData_, "hs1", point_hs1);
		dataSetFieldAdd.AddPointField(vtkmData_, "hs2", point_hs2);
		dataSetFieldAdd.AddPointField(vtkmData_, "hs3", point_hs3);
		dataSetFieldAdd.AddPointField(vtkmData_, "hs", point_hs);

		// vtkm::io::writer::VTKDataSetWriter writer("tetrahedron.vtk");
		// writer.WriteDataSet(vtkmData_);
		// core::utility::log::Log::DefaultLog.WriteInfo("vtkmData_ is successfully stored in tetrahedron.vtk.");

		// get min max bounds from dataset
        minMaxBounds_ = vtkmData_.GetCoordinateSystem(0).GetBounds();

		lhsVtkmDc->UpdateDataChanges(true);
        lhsVtkmDc->SetDataSet(&this->vtkmData_);

        return true;
    }


	lhsVtkmDc->UpdateDataChanges(false);

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

	size_t newNodeHash = rhsTopNodesCad->getDataHash();
    size_t newLabelHash = rhsBottomLabelCad->getDataHash();


	bool update = newNodeHash != oldNodeDataHash_ || newLabelHash != oldLabelDataHash_;

	if (update) {
        lhsVtkmDc->SetBounds(this->minMaxBounds_);

		oldNodeDataHash_ = newNodeHash;
		oldLabelDataHash_ = newLabelHash;
		return true;
    }

    return true;
}
