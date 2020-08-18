/*
 * mmvtkmDataSource.cpp (MMPLDDataSource)
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmvtkm/mmvtkmDataSource.h"
#include "CallADIOSData.cpp"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
//#include "vtkm/io/reader/VTKPolyDataReader.h"
#include "vtkm/cont/DataSetBuilderExplicit.h"
#include "vtkm/cont/DataSetFieldAdd.h"
#include "vtkm/io/reader/VTKDataSetReader.h"
#include "vtkm/io/writer/VTKDataSetWriter.h"
#include "vtkm/Math.h"
#include "vtkm/Matrix.h"


using namespace megamol;
using namespace megamol::mmvtkm;

typedef vtkm::Vec<float, 3> Vec3f;

/*
 * moldyn::mmvtkmDataSource::mmvtkmDataSource
 */
mmvtkmDataSource::mmvtkmDataSource(void)
    : core::view::AnimDataModule()
    , getData("getdata", "Slot to request data from this data source.")
    , nodesAdiosCallerSlot("adiosNodeSlot", "Slot to request node data from adios.")
    , labelAdiosCallerSlot("adiosLabelSlot", "Slot to request label data from adios.")
    , filename("filename", "The path to the vtkm file to load.")
    , data_hash(0)
    , oldNodeDataHash(0) 
	, oldLabelDataHash(0)
    , vtkmDataFile("")
    , vtkmData()
    , minMaxBounds(0.f, 0.f, 0.f, 0.f, 0.f, 0.f) 
{

	// CURRENTLY NOT USED
    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&mmvtkmDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(0),
        &mmvtkmDataSource::getDataCallback); // GetData is FunctionName(0)
    this->getData.SetCallback(mmvtkmDataCall::ClassName(), mmvtkmDataCall::FunctionName(1),
        &mmvtkmDataSource::getMetaDataCallback); // GetExtent is FunctionName(1)
    this->MakeSlotAvailable(&this->getData);

    this->nodesAdiosCallerSlot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->nodesAdiosCallerSlot);

    this->labelAdiosCallerSlot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->labelAdiosCallerSlot);
}


/*
 * moldyn::mmvtkmDataSource::~mmvtkmDataSource
 */
mmvtkmDataSource::~mmvtkmDataSource(void) { this->Release(); }
/*
 moldyn::mmvtkmDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* mmvtkmDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<mmvtkmDataSource*>(this));
    return f;
}


/*
 * moldyn::mmvtkmDataSource::create
 */
bool mmvtkmDataSource::create(void) { return true; }

/*
 * moldyn::mmvtkmDataSource::loadFrame
 */
void mmvtkmDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
}


/*
 * moldyn::mmvtkmDataSource::release
 */
void mmvtkmDataSource::release(void) {
}

/*
 * moldyn::mmvtkmDataSource::filenameChanged
 */
bool mmvtkmDataSource::filenameChanged(core::param::ParamSlot& slot) {
    vtkmDataFile = this->filename.Param<core::param::FilePathParam>()->ValueString();
    if (vtkmDataFile.empty()) {
        vislib::sys::Log::DefaultLog.WriteInfo("Empty vtkm file!");
    }

    vislib::sys::Log::DefaultLog.WriteInfo(
        "If no \"Safety check\" is shown, something went wrong reading the data. Probably the necessary line "
        "is not commented out. See readme");

    vtkm::io::reader::VTKDataSetReader readData(vtkmDataFile);
    vtkmData = readData.ReadDataSet();
    vislib::sys::Log::DefaultLog.WriteInfo("Safety check");

    return true;
}


/*
 * moldyn::mmvtkmDataSource::getDataCallback
 */
bool mmvtkmDataSource::getDataCallback(core::Call& caller) {
    mmvtkmDataCall* c2 = dynamic_cast<mmvtkmDataCall*>(&caller);
    if (c2 == NULL) return false;

    adios::CallADIOSData* nodesCad = this->nodesAdiosCallerSlot.CallAs<adios::CallADIOSData>();
    if (nodesCad == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("nodesCallADIOSData is nullptr");
        return false;
    }

    adios::CallADIOSData* labelCad = this->labelAdiosCallerSlot.CallAs<adios::CallADIOSData>();
    if (labelCad == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("labelCallADIOSData is nullptr");
        return false;
    }

	// getDataCallback
    if (!(*nodesCad)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during nodes getData");
        return false;
    }

    if (!(*labelCad)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during label getData");
        return false;
    }

	size_t newNodeHash = nodesCad->getDataHash();		// TODO how hash is set in tabletoadios or csvdatasource
	size_t newLabelHash = labelCad->getDataHash();
    bool update = newNodeHash != oldNodeDataHash || newLabelHash != oldLabelDataHash;


    if (update) {
		/////////////////////////////
		// get tetrahedron coordinates from csv file through adios
		/////////////////////////////
		std::string coord_type = nodesCad->getData("x")->getType();
		std::string label_type = nodesCad->getData("node-label")->getType();

		// this assumes the data type from the file is float
		// TODO do for general case
		std::vector<float> node_labels = nodesCad->getData("node-label")->GetAsFloat();
		std::vector<float> x_coord = nodesCad->getData("x")->GetAsFloat();
		std::vector<float> y_coord = nodesCad->getData("y")->GetAsFloat();
		std::vector<float> z_coord = nodesCad->getData("z")->GetAsFloat();
		std::vector<float> s11 = nodesCad->getData("s11")->GetAsFloat();
		std::vector<float> s12 = nodesCad->getData("s12")->GetAsFloat();
		std::vector<float> s13 = nodesCad->getData("s13")->GetAsFloat();
		std::vector<float> s21 = s12;
		std::vector<float> s22 = nodesCad->getData("s22")->GetAsFloat();
		std::vector<float> s23 = nodesCad->getData("s23")->GetAsFloat();
		std::vector<float> s31 = s13;
		std::vector<float> s32 = s23;
		std::vector<float> s33 = nodesCad->getData("s33")->GetAsFloat();
		std::vector<float> hs1_wert = nodesCad->getData("hs1_wert")->GetAsFloat();
		std::vector<float> hs1_x = nodesCad->getData("hs1_x")->GetAsFloat();
		std::vector<float> hs1_y = nodesCad->getData("hs1_y")->GetAsFloat();
		std::vector<float> hs1_z = nodesCad->getData("hs1_z")->GetAsFloat();
		std::vector<float> hs2_wert = nodesCad->getData("hs2_wert")->GetAsFloat();
		std::vector<float> hs2_x = nodesCad->getData("hs2_x")->GetAsFloat();
		std::vector<float> hs2_y = nodesCad->getData("hs2_y")->GetAsFloat();
		std::vector<float> hs2_z = nodesCad->getData("hs2_z")->GetAsFloat();
		std::vector<float> hs3_wert = nodesCad->getData("hs3_wert")->GetAsFloat();
		std::vector<float> hs3_x = nodesCad->getData("hs3_x")->GetAsFloat();
		std::vector<float> hs3_y = nodesCad->getData("hs3_y")->GetAsFloat();
		std::vector<float> hs3_z = nodesCad->getData("hs3_z")->GetAsFloat();

		// TODO check if size of each vector is the same (assert)
		std::vector<float> element_label = labelCad->getData("element-label")->GetAsFloat();
		std::vector<float> nodeA = labelCad->getData("nodea")->GetAsFloat();
		std::vector<float> nodeB = labelCad->getData("nodeb")->GetAsFloat();
		std::vector<float> nodeC = labelCad->getData("nodec")->GetAsFloat();
		std::vector<float> nodeD = labelCad->getData("noded")->GetAsFloat();


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
		            // vislib::sys::Log::DefaultLog.WriteInfo("(%i, %i) with %i", i, j, labels[j]);
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

		vislib::sys::Log::DefaultLog.WriteInfo("Number of skipped tetrahedrons: %i", num_skipped);

		vtkmData = dataSetBuilder.Create(points, cell_shapes, num_indices, cell_indices);
		dataSetFieldAdd.AddPointField(vtkmData, "hs1", point_hs1);
		dataSetFieldAdd.AddPointField(vtkmData, "hs2", point_hs2);
		dataSetFieldAdd.AddPointField(vtkmData, "hs3", point_hs3);
		dataSetFieldAdd.AddPointField(vtkmData, "hs", point_hs);

		vtkm::io::writer::VTKDataSetWriter writer("tetrahedron.vtk");
		writer.WriteDataSet(vtkmData);
		vislib::sys::Log::DefaultLog.WriteInfo("vtkmData is successfully stored in tetrahedron.vtk.");

		// get min max bounds from dataset
        minMaxBounds = vtkmData.GetCoordinateSystem(0).GetBounds();

		c2->UpdateDataChanges(true);
        c2->SetDataSet(&this->vtkmData);

        return true;
    }


	c2->UpdateDataChanges(false);

    return true;
}


/*
 * moldyn::mmvtkmDataSource::getMetaDataCallback
 */
bool mmvtkmDataSource::getMetaDataCallback(core::Call& caller) {
    mmvtkmDataCall* c2 = dynamic_cast<mmvtkmDataCall*>(&caller);

	adios::CallADIOSData* nodesCad = this->nodesAdiosCallerSlot.CallAs<adios::CallADIOSData>();
    if (nodesCad == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("nodesCallADIOSData is nullptr");
        return false;
    }

    adios::CallADIOSData* labelCad = this->labelAdiosCallerSlot.CallAs<adios::CallADIOSData>();
    if (labelCad == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("labelCallADIOSData is nullptr");
        return false;
    }

	size_t newNodeHash = nodesCad->getDataHash();
    size_t newLabelHash = labelCad->getDataHash();

	bool update = newNodeHash != oldNodeDataHash || newLabelHash != oldLabelDataHash;

    c2->UpdateDataChanges(update);

	if (update) {
        c2->SetBounds(this->minMaxBounds);

		oldNodeDataHash = newNodeHash;
		oldLabelDataHash = newLabelHash;
		return true;
    }

    return true;
}
