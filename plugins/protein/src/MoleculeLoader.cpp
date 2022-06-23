/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "MoleculeLoader.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"

using namespace megamol::core;
using namespace megamol::protein;

MoleculeLoader::MoleculeLoader(void)
        : core::Module()
        , data_out_slot_("dataOut", "Connects the loader with the requesting modules")
        , filename_slot_("filename", "Path to the file containing the structure of the molecule(s) to visualize.")
        , trajectory_filename_slot_("trajectoryFilename", "Path to the file containing the trajectory of the molecule.")
        , structure_(nullptr)
        , trajectory_(nullptr) {
    // Callee slots
    data_out_slot_.SetCallback(protein_calls::MolecularDataCall::ClassName(),
        protein_calls::MolecularDataCall::FunctionName(protein_calls::MolecularDataCall::CallForGetData),
        &MoleculeLoader::getMDCData);
    data_out_slot_.SetCallback(protein_calls::MolecularDataCall::ClassName(),
        protein_calls::MolecularDataCall::FunctionName(protein_calls::MolecularDataCall::CallForGetExtent),
        &MoleculeLoader::getMDCExtent);
    // TODO connect slots for the new call
    this->MakeSlotAvailable(&data_out_slot_);

    // Parameter slots
    filename_slot_.SetParameter(
        new param::FilePathParam("", param::FilePathParam::FilePathFlags_::Flag_Any_ToBeCreated));
    this->MakeSlotAvailable(&filename_slot_);

    trajectory_filename_slot_.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&trajectory_filename_slot_);
}

MoleculeLoader::~MoleculeLoader(void) {
    this->Release();
}

bool MoleculeLoader::create(void) {
    // TODO

    return true;
}

void MoleculeLoader::release(void) {
    // TODO
}

bool MoleculeLoader::getMDCExtent(core::Call& call) {
    auto mdc = dynamic_cast<protein_calls::MolecularDataCall*>(&call);
    if (mdc == nullptr) {
        return false;
    }

    updateFiles();
    // TODO

    return true;
}

bool MoleculeLoader::getMDCData(core::Call& call) {
    auto mdc = dynamic_cast<protein_calls::MolecularDataCall*>(&call);
    if (mdc == nullptr) {
        return false;
    }

    updateFiles();
    // TODO

    return true;
}

bool MoleculeLoader::getMoleculeExtent(core::Call& call) {
    // TODO
    return true;
}

bool MoleculeLoader::getMoleculeData(core::Call& call) {
    // TODO
    return true;
}

bool MoleculeLoader::loadFile(std::filesystem::path const& path_to_structure) {
    return loadFile(path_to_structure, path_to_structure);
}

bool MoleculeLoader::loadFile(
    std::filesystem::path const& path_to_structure, std::filesystem::path const& path_to_trajectory) {
    if (!path_to_structure.empty()) {
        structure_ = std::make_shared<chemfiles::Trajectory>(path_to_structure.string());
    } else {
        return false;
    }
    if (path_to_structure == path_to_trajectory) {
        trajectory_ = structure_;
        return true;
    }
    if (!path_to_trajectory.empty()) {
        trajectory_ = std::make_shared<chemfiles::Trajectory>(path_to_trajectory.string());
    } else {
        return false;
    }
    return true;
}

void MoleculeLoader::updateFiles() {
    if (filename_slot_.IsDirty() || trajectory_filename_slot_.IsDirty()) {
        filename_slot_.ResetDirty();
        trajectory_filename_slot_.ResetDirty();
        auto const filename = filename_slot_.Param<param::FilePathParam>()->Value();
        auto const traj_filename = trajectory_filename_slot_.Param<param::FilePathParam>()->Value();
        bool change = false;
        if (filename != path_to_current_structure_) {
            change = true;
        }
        if (traj_filename != path_to_current_trajectory_) {
            change = true;
        }
        if (change) {
            bool success = false;
            if (filename == traj_filename) {
                success = loadFile(filename);
            } else {
                success = loadFile(filename, traj_filename);
            }
            if (success) {
                path_to_current_structure_ = filename;
                path_to_current_trajectory_ = traj_filename;
            }
        }
    }
}

void MoleculeLoader::postProcessFilesMDC() {
    if (structure_ == nullptr) {
        return;
    }
    auto structure = structure_->read();
}

void MoleculeLoader::postPorcessFilesMolecule() {
    // TODO
}
