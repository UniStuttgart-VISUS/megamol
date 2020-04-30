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
    , vtkmDataFile("")
    , vtkmData() {

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
    this->data_hash++;

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

    /////////////////////////////
    // get tetrahedron coordinates from csv file through adios
    /////////////////////////////

    // TODO: look what happens detailed
    // TODO: place it somewhere else? or check if data has changed from callee
    // getHeaderCallback
    if (!(*nodesCad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during nodes getExtents");
        return false;
    }

    if (!(*labelCad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Error during label getExtents");
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


    std::string coord_type = nodesCad->getData("x")->getType();
    std::string label_type = nodesCad->getData("node-label")->getType();

    // this assumes the data type from the file is float
    // TODO do for general case
    std::vector<float> node_labels = nodesCad->getData("node-label")->GetAsFloat();
    std::vector<float> x_coord = nodesCad->getData("x")->GetAsFloat();
    std::vector<float> y_coord = nodesCad->getData("y")->GetAsFloat();
    std::vector<float> z_coord = nodesCad->getData("z")->GetAsFloat();
    std::vector<float> s11 = nodesCad->getData("s11")->GetAsFloat();
    std::vector<float> s12 = nodesCad->getData("s11")->GetAsFloat();
    std::vector<float> s13 = nodesCad->getData("s11")->GetAsFloat();
    std::vector<float> s21 = s12;
    std::vector<float> s22 = nodesCad->getData("s11")->GetAsFloat();
    std::vector<float> s23 = nodesCad->getData("s11")->GetAsFloat();
    std::vector<float> s31 = s13;
    std::vector<float> s32 = s23;
    std::vector<float> s33 = nodesCad->getData("s11")->GetAsFloat();
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
    std::vector<vtkm::Vec3f_32> points(num_nodes);
    std::vector<vtkm::Vec3f_32> point_hs1(num_nodes);
    std::vector<vtkm::Vec3f_32> point_hs2(num_nodes);
    std::vector<vtkm::Vec3f_32> point_hs3(num_nodes);
    std::vector<vtkm::Vec3f_32> point_hs(num_nodes);
    std::vector<vtkm::Matrix<vtkm::Float32, 3, 3>> point_tensors(num_nodes);
    //std::vector<vtkm::Id> cell_indices(num_elements * 4);
    std::vector<vtkm::Id> cell_indices;
    std::vector<vtkm::IdComponent> num_indices;
    std::vector<vtkm::UInt8> cell_shapes;


	for (int i = 0; i < num_nodes; ++i) {
        vtkm::Vec3f_32 point = {x_coord[i], y_coord[i], z_coord[i]};
        vtkm::Vec3f_32 hs1 = {hs1_x[i], hs1_y[i], hs1_z[i]};
        vtkm::Vec3f_32 hs2 = {hs2_x[i], hs2_y[i], hs2_z[i]};
        vtkm::Vec3f_32 hs3 = {hs3_x[i], hs3_y[i], hs3_z[i]};

		vtkm::Matrix<vtkm::Float32, 3, 3> tensor;
		point_tensors[0][0] = s11[i];
		point_tensors[0][1] = s12[i];
		point_tensors[0][2] = s13[i];
		point_tensors[1][0] = s21[i];
		point_tensors[1][1] = s22[i];
		point_tensors[1][2] = s23[i];
		point_tensors[2][0] = s31[i];
		point_tensors[2][1] = s32[i];
		point_tensors[2][2] = s33[i];

		vtkm::Vec3f_32 hs1_tens_prod = hs1_wert[i] * vtkm::MatrixMultiply(tensor, hs1);
        vtkm::Vec3f_32 hs2_tens_prod = hs2_wert[i] * vtkm::MatrixMultiply(tensor, hs2);
        vtkm::Vec3f_32 hs3_tens_prod = hs3_wert[i] * vtkm::MatrixMultiply(tensor, hs3);

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
            int idx = 3 * j;
            index_buffer[j] = {node_index};
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

    this->data_hash++;

    this->filename.ForceSetDirty();
    // update data only when we have a new file
    if (this->filename.IsDirty()) {
        c2->SetDataHash(this->data_hash);
        c2->SetDataSet(&vtkmData);
        filename.ResetDirty();

        return true;
    }

    return false;
}


/*
 * moldyn::mmvtkmDataSource::getMetaDataCallback
 */
bool mmvtkmDataSource::getMetaDataCallback(core::Call& caller) {
    mmvtkmDataCall* c2 = dynamic_cast<mmvtkmDataCall*>(&caller);

    if (c2 != NULL) {
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}
