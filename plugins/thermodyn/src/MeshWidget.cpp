#include "MeshWidget.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "mesh/MeshDataAccessor.h"


megamol::thermodyn::MeshWidget::MeshWidget()
        : in_data_slot_("dataIn", "")
        , in_info_slot_("infoIn", "")
        , flags_read_slot_("flagsRead", "")
        , accumulate_slot_("accumulate", "") {
    in_data_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&in_data_slot_);

    in_info_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_info_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);

    accumulate_slot_ << new core::param::BoolParam(false);
    MakeSlotAvailable(&accumulate_slot_);
}


megamol::thermodyn::MeshWidget::~MeshWidget() {
    this->Release();
}


bool megamol::thermodyn::MeshWidget::create() {
    ctx_ = ImPlot::CreateContext();

    return true;
}


void megamol::thermodyn::MeshWidget::release() {
    ImPlot::DestroyContext(ctx_);
}


bool megamol::thermodyn::MeshWidget::Render(core::view::CallRender3DGL& call) {
    auto mesh_in = in_data_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto info_in = in_info_slot_.CallAs<core::moldyn::MultiParticleDataCall>();

    auto meta = mesh_in->getMetaData();
    meta.m_frame_ID = call.Time();
    if (!(*mesh_in)(0))
        return false;

    if (info_in) {
        info_in->SetFrameID(call.Time());
        if (!(*info_in)(0))
            return false;
    }

    auto flags_read = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (flags_read == nullptr)
        return false;

    if (!(*flags_read)(0))
        return false;

    parse_data(*mesh_in, info_in, *flags_read);

    return true;
}


bool megamol::thermodyn::MeshWidget::GetExtents(core::view::CallRender3DGL& call) {
    auto mesh_in = in_data_slot_.CallAs<mesh::CallMesh>();
    if (mesh_in == nullptr)
        return false;

    auto info_in = in_info_slot_.CallAs<core::moldyn::MultiParticleDataCall>();

    auto meta = mesh_in->getMetaData();
    meta.m_frame_ID = call.Time();
    if (!(*mesh_in)(1))
        return false;
    meta = mesh_in->getMetaData();

    if (info_in) {
        info_in->SetFrameID(call.Time());
        if (!(*info_in)(1))
            return false;
    }

    call.SetTimeFramesCount(meta.m_frame_cnt);

    return true;
}


bool megamol::thermodyn::MeshWidget::widget(float x, float y, std::size_t idx,
    mesh::MeshDataAccessCollection::Mesh const& mesh, core::moldyn::SimpleSphericalParticles const* info) {
    ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Appearing);

    // ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Appearing);

    bool plot_open = true;
    ImGui::Begin((std::string("Test Plot ") + std::to_string(idx)).c_str(), &plot_open,
        ImGuiWindowFlags_NoTitleBar); // | ImGuiWindowFlags_NoBackground);

    /*if (ImPlot::BeginPlot("data", nullptr, nullptr, ImVec2(-1, 0), ImPlotFlags_Query)) {
        ImPlot::EndPlot();
    }*/

    auto const mesh_acc = mesh::MeshDataTriangleAccessor(mesh);

    ImGui::Text("Triangle ID %d", idx);
    // ImGui::Text("ICol Val %f", ic_acc->Get_f(idx));
    if (info) {
        auto const indices = mesh_acc.GetIndices(idx);

        auto const i_acc = info->GetParticleStore().GetCRAcc();

        ImGui::Text("Info %f %f %f", i_acc->Get_f(indices.x), i_acc->Get_f(indices.y), i_acc->Get_f(indices.z));
    }

    ImGui::End();

    return true;
}


bool megamol::thermodyn::MeshWidget::widget(float x, float y,
    std::list<std::pair<std::size_t, mesh::MeshDataAccessCollection::Mesh const*>> const& selected,
    core::moldyn::SimpleSphericalParticles const* info) {
    ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Appearing);

    // ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Appearing);

    bool plot_open = true;
    ImGui::Begin((std::string("Test Plot ")).c_str(), &plot_open,
        ImGuiWindowFlags_NoTitleBar); // | ImGuiWindowFlags_NoBackground);

    /*if (ImPlot::BeginPlot("data", nullptr, nullptr, ImVec2(-1, 0), ImPlotFlags_Query)) {
        ImPlot::EndPlot();
    }*/

    // ImGui::Text("ICol Val %f", ic_acc->Get_f(idx));
    if (info) {
        std::list<float> values;
        for (auto const& [idx, mesh] : selected) {
            auto const mesh_acc = mesh::MeshDataTriangleAccessor(*mesh);
            auto const indices = mesh_acc.GetIndices(idx);
            auto const i_acc = info->GetParticleStore().GetCRAcc();
            values.push_back(i_acc->Get_f(indices.x));
            values.push_back(i_acc->Get_f(indices.y));
            values.push_back(i_acc->Get_f(indices.z));
        }

        float avg_val = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (auto const& el : values) {
            avg_val += el;
            if (min_val > el)
                min_val = el;
            if (max_val < el)
                max_val = el;
        }
        avg_val /= static_cast<float>(values.size());

        ImGui::Text("Avg Val %f", avg_val);
        ImGui::Text("Min Val %f Max Val %f", min_val, max_val);
    }

    ImGui::End();

    return true;
}


bool megamol::thermodyn::MeshWidget::parse_data(
    mesh::CallMesh& in_mesh, core::moldyn::MultiParticleDataCall* in_info, core::FlagCallRead_CPU& fcr) {
    auto const& meshes = in_mesh.getData()->accessMeshes();
    std::vector<std::string> mesh_names;
    mesh_names.reserve(meshes.size());
    mesh_prefix_count_.resize(meshes.size());
    auto counter = 0u;
    for (auto const& entry : meshes) {
        mesh_names.push_back(entry.first);
        auto c_count = 0;
        switch (entry.second.primitive_type) {
        case mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES: {
            c_count = 3;
        } break;
        case mesh::MeshDataAccessCollection::PrimitiveType::QUADS: {
            c_count = 4;
        } break;
        }
        if (c_count == 0) {
            mesh_prefix_count_[counter] = counter == 0 ? 0 : mesh_prefix_count_[counter - 1];
            ++counter;
            break;
        }
        auto const c_bs = mesh::MeshDataAccessCollection::getByteSize(entry.second.indices.type);
        auto const num_el = entry.second.indices.byte_size / (c_bs * c_count);
        mesh_prefix_count_[counter] = counter == 0 ? num_el : mesh_prefix_count_[counter - 1] + num_el;
        ++counter;
    }

    auto const selection_data = fcr.getData();

    core::moldyn::SimpleSphericalParticles* info = nullptr;

    if (in_info) {
        info = &(in_info->AccessParticles(0));
    }

    if (accumulate_slot_.Param<core::param::BoolParam>()->Value() && in_info) {
        std::list<std::pair<std::size_t, mesh::MeshDataAccessCollection::Mesh const*>> selected;
        for (decltype(selection_data->flags)::element_type::size_type i = 0; i < selection_data->flags->size(); ++i) {
            auto const el = (*selection_data->flags)[i];
            if (el == core::FlagStorage::SELECTED) {
                auto const fit =
                    std::find_if(mesh_prefix_count_.begin(), mesh_prefix_count_.end(), [i](auto el) { return el > i; });
                if (fit != mesh_prefix_count_.end()) {
                    auto const mesh_idx = std::distance(mesh_prefix_count_.begin(), fit);
                    auto idx = i;
                    if (mesh_idx != 0)
                        idx -= *fit;
                    selected.push_back(std::make_pair(idx, &(meshes.at(mesh_names[mesh_idx]))));
                }
            }
        }
        if (!selected.empty())
            widget(mouse_x_, mouse_y_, selected, info);
    } else {
        for (decltype(selection_data->flags)::element_type::size_type i = 0; i < selection_data->flags->size(); ++i) {
            auto const el = (*selection_data->flags)[i];
            if (el == core::FlagStorage::SELECTED) {
                auto const fit =
                    std::find_if(mesh_prefix_count_.begin(), mesh_prefix_count_.end(), [i](auto el) { return el > i; });
                if (fit != mesh_prefix_count_.end()) {
                    auto const mesh_idx = std::distance(mesh_prefix_count_.begin(), fit);
                    auto idx = i;
                    if (mesh_idx != 0)
                        idx -= *fit;
                    widget(mouse_x_, mouse_y_, idx, meshes.at(mesh_names[mesh_idx]), info);
                }
            }
        }
    }

    return true;
}


bool megamol::thermodyn::MeshWidget::OnMouseMove(double x, double y) {
    mouse_x_ = static_cast<float>(x);
    mouse_y_ = static_cast<float>(y);
    return false;
}
