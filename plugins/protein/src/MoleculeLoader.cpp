/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "MoleculeLoader.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"

#include <iostream>

using namespace megamol::core;
using namespace megamol::protein;

MoleculeLoader::MoleculeLoader(void)
        : core::Module()
        , data_out_slot_("dataOut", "Connects the loader with the requesting modules")
        , filename_slot_("filename", "Path to the file containing the structure of the molecule(s) to visualize.")
        , trajectory_filename_slot_("trajectoryFilename", "Path to the file containing the trajectory of the molecule.")
        , calc_secstruct_slot_(
              "calcSecstruct", "Enables the calculation of the Secondary Structure using the STRIDE algorithm")
        , used_radius_slot_("radiusMeasure", "Selection for the radius measure used for the atoms")
        , structure_(nullptr)
        , time_step_count_(0) {
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

    calc_secstruct_slot_.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&calc_secstruct_slot_);

    auto const en = new param::EnumParam(0);
    en->SetTypePair(0, "Van der Waals");
    en->SetTypePair(1, "Covalent");
    en->SetTypePair(2, "Constant");
    used_radius_slot_.SetParameter(en);
    this->MakeSlotAvailable(&used_radius_slot_);

    global_bounding_box_.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    local_bounding_box_ = global_bounding_box_;
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
    bool const recalc = updateFiles();
    postProcessFilesMDC(*mdc, recalc);
    if (structure_ != nullptr) {
        mdc->SetFrameCount(std::max(time_step_count_, static_cast<int64_t>(1)));
    } else {
        mdc->SetFrameCount(1);
    }
    mdc->AccessBoundingBoxes().SetObjectSpaceBBox(global_bounding_box_);
    mdc->AccessBoundingBoxes().SetObjectSpaceClipBox(global_bounding_box_);

    return true;
}

bool MoleculeLoader::getMDCData(core::Call& call) {
    auto mdc = dynamic_cast<protein_calls::MolecularDataCall*>(&call);
    if (mdc == nullptr) {
        return false;
    }
    bool const recalc = updateFiles();
    postProcessFilesMDC(*mdc, recalc);

    auto const& s = mdc_structures_;
    auto const& f = mdc_structures_frame_;
    mdc->SetAtoms(f.atom_positions_.size(), s.atom_types_.size(), s.atom_type_indices_.data(),
        &f.atom_positions_.front().x, s.atom_types_.data(), s.atom_residue_indices_.data(), f.b_factors_.data(),
        f.charges_.data(), f.occupancies_.data());
    mdc->SetBFactorRange(f.b_factor_bounds_.first, f.b_factor_bounds_.second);
    mdc->SetOccupancyRange(f.occupancy_bounds_.first, f.occupancy_bounds_.second);
    mdc->SetChargeRange(f.charge_bounds_.first, f.charge_bounds_.second);
    mdc->SetFormerAtomIndices(s.former_atom_indices_.data());

    mdc->SetConnections(s.connectivity_.size(), &s.connectivity_.front().x);
    mdc->SetResidues(s.residues_.size(), (const protein_calls::MolecularDataCall::Residue**)s.residues_.data());
    mdc->SetSolventResidueIndices(s.solvent_residue_index_.size(), s.solvent_residue_index_.data());
    mdc->SetResidueTypeNames(s.residue_type_names_.size(), s.residue_type_names_.data());
    mdc->SetMolecules(s.molecules_.size(), s.molecules_.data());
    mdc->SetChains(s.chains_.size(), s.chains_.data());
    mdc->SetFilter(s.visibility_per_atom_.data());
    // TODO secondary structure

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
    if (!path_to_structure.empty() && path_to_trajectory.empty()) {
        structure_ = std::make_shared<chemfiles::Trajectory>(path_to_structure.string());
    } else if (!path_to_structure.empty() && !path_to_trajectory.empty()) {
        if (path_to_structure == path_to_trajectory) {
            structure_ = std::make_shared<chemfiles::Trajectory>(path_to_structure.string());
        } else {
            structure_ = std::make_shared<chemfiles::Trajectory>(path_to_trajectory.string());
            structure_->set_topology(path_to_structure.string());
        }
    } else {
        utility::log::Log::DefaultLog.WriteError("The paths to properly load a molecule is missing");
        return false;
    }
    return true;
}

bool MoleculeLoader::updateFiles() {
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
                time_step_count_ = 1;
            } else {
                success = loadFile(filename, traj_filename);
                time_step_count_ = structure_->nsteps();
            }
            if (success) {
                path_to_current_structure_ = filename;
                path_to_current_trajectory_ = traj_filename;
            }
        }
        return true;
    }
    return false;
}

void MoleculeLoader::postProcessFilesMDC(protein_calls::MolecularDataCall& mdc, bool recalc) {
    if (structure_ == nullptr) {
        return;
    }

    if (recalc) {
        auto protein = structure_->read_step(static_cast<int>(mdc.Calltime()) % time_step_count_);
        auto const atom_count = protein.topology().size();
        auto const residue_count = protein.topology().residues().size();
        auto const connection_count = protein.topology().bonds().size();

        if (atom_count > 0) {
            // delete old data
            mdc_structures_ = {};
            mdc_structures_frame_ = {};

            // calc bounding box
            auto const& first_atom = protein.topology()[0];
            auto const& first_pos = protein.positions().at(0);
            auto const first_vdw_radius = first_atom.vdw_radius().value_or(0.0f);
            local_bounding_box_.Set(first_pos[0] - first_vdw_radius, first_pos[1] - first_vdw_radius,
                first_pos[2] - first_vdw_radius, first_pos[0] + first_vdw_radius, first_pos[1] + first_vdw_radius,
                first_pos[2] + first_vdw_radius);
            for (size_t i = 0; i < atom_count; ++i) {
                auto const& atom = protein.topology()[i];
                auto const& atom_pos = protein.positions()[i];
                auto const vdw_radius = atom.vdw_radius().value_or(0.0);
                vislib::math::Cuboid<float> const atom_bbox(atom_pos[0] - vdw_radius, atom_pos[1] - vdw_radius,
                    atom_pos[2] - vdw_radius, atom_pos[0] + vdw_radius, atom_pos[1] + vdw_radius,
                    atom_pos[2] + vdw_radius);
                local_bounding_box_.Union(atom_bbox);
            }
            // TODO handle per-frame bounding boxes
            global_bounding_box_ = local_bounding_box_;
            // TODO consider bounding boxes stored by chemfiles

            // copy over atom positions
            mdc_structures_frame_.atom_positions_.resize(atom_count);
            for (size_t i = 0; i < atom_count; ++i) {
                mdc_structures_frame_.atom_positions_[i] = glm::make_vec3(&protein.positions().at(i)[0]);
            }
            // TODO charge, bfactor, occupancy
            mdc_structures_frame_.b_factors_.resize(atom_count, 0.0f);
            mdc_structures_frame_.occupancies_.resize(atom_count, 1.0f);
            mdc_structures_frame_.charges_.resize(atom_count, 0.0f);
            mdc_structures_frame_.b_factor_bounds_ = std::make_pair(0.0f, 1.0);
            mdc_structures_frame_.occupancy_bounds_ = std::make_pair(0.0f, 1.0);
            mdc_structures_frame_.charge_bounds_ =
                std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());

            mdc_structures_.atom_type_indices_.resize(atom_count, 0);
            mdc_structures_.former_atom_indices_.resize(atom_count, 0);
            mdc_structures_.atom_residue_indices_.resize(atom_count, 0);
            // TODO atom_types_
            mdc_structures_.residues_.resize(residue_count, nullptr);
            mdc_structures_.residue_type_names_.resize(residue_count, "");
            // TODO solvent_residue_index
            // TODO molecules
            // TODO chains
            mdc_structures_.connectivity_.resize(connection_count, glm::vec2(0, 0));
            mdc_structures_.visibility_per_atom_.resize(atom_count, 1);

            /*
             * short naming guide:
             *  - .full_name() is the human-readable name of the element. Should not be used
             *  - .name() is the name of the specific atom, so a C-alpha atom will have the name "CA"
             *  - .type() is the element symbol of the atom, so a C-alpha atom will have the type "C"
             *
             * spaces are removed beforehand
             */

            for (size_t i = 0; i < atom_count; ++i) {
                auto const& cur_atom = protein.topology()[i];
                // charge
                mdc_structures_frame_.charges_[i] = cur_atom.charge();
                mdc_structures_frame_.charge_bounds_.first =
                    cur_atom.charge() < mdc_structures_frame_.charge_bounds_.first
                        ? cur_atom.charge()
                        : mdc_structures_frame_.charge_bounds_.first;
                mdc_structures_frame_.charge_bounds_.second =
                    cur_atom.charge() < mdc_structures_frame_.charge_bounds_.second
                        ? cur_atom.charge()
                        : mdc_structures_frame_.charge_bounds_.second;
            }
        }
    }
}

void MoleculeLoader::postProcessFilesMolecule() {
    // TODO
}
