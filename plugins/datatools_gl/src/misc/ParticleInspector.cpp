#include "ParticleInspector.h"

#include <imgui.h>
#include <imgui_internal.h>

#include "FrameStatistics.h"

using namespace megamol::datatools_gl::misc;
using namespace megamol;

/*
 * megamol::moldyn_gl::misc::ParticleInspector::ParticleInspector
 */
ParticleInspector::ParticleInspector()
        : Module()
        , slotParticlesOut("particlesout", "Provides the particles.")
        , slotParticlesIn("particlesin", "Ingests the particles.") {

    /* Register parameters. */

    /* Register calls. */
    this->slotParticlesOut.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &ParticleInspector::getParticleData);
    this->slotParticlesOut.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &ParticleInspector::getParticleExtents);

    this->MakeSlotAvailable(&this->slotParticlesOut);

    this->slotParticlesIn.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->slotParticlesIn);
}


/*
 * megamol::moldyn_gl::misc::ParticleInspector::~ParticleInspector
 */
ParticleInspector::~ParticleInspector() {
    this->Release();
}


/*
 * megamol::moldyn_gl::misc::ParticleInspector::create
 */
bool ParticleInspector::create() {
    return true;
}

/*
 * megamol::moldyn_gl::misc::ParticleInspector::getParticleData
 */
bool ParticleInspector::getParticleData(core::Call& call) {
    auto* p_in = this->slotParticlesIn.CallAs<geocalls::MultiParticleDataCall>();

    auto* p_out = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (p_out == nullptr)
        return false;

    if (p_in != nullptr) {
        *p_in = *p_out;
        if (!(*p_in)(0))
            return false;
        *p_out = *p_in;
        if (p_in->GetParticleListCount() > 0) {
            drawTable(p_in);
        }
        p_in->SetUnlocker(nullptr, false);
        p_in->Unlock();
    } else {
        return false;
    }
    return true;
}


/*
 * megamol::moldyn_gl::misc::ParticleInspector::getParticleExtents
 */
bool ParticleInspector::getParticleExtents(core::Call& call) {
    auto* p_in = this->slotParticlesIn.CallAs<geocalls::MultiParticleDataCall>();

    auto* p_out = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (p_out == nullptr)
        return false;

    if (p_in != nullptr) {
        *p_in = *p_out;
        if (!(*p_in)(1))
            return false;
        *p_out = *p_in;
        p_in->SetUnlocker(nullptr, false);
        p_in->Unlock();
    } else {
        return false;
    }
    return true;
}


/*
 * megamol::moldyn_gl::misc::ParticleInspector::release
 */
void ParticleInspector::release() {}


/*
 * megamol::moldyn_gl::misc::ParticleInspector::drawTable
 */
void ParticleInspector::drawTable(geocalls::MultiParticleDataCall* p_in) {
    bool valid_imgui_scope =
        ((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false));
    if (!valid_imgui_scope)
        return;

    auto current_frame = frontend_resources.get<frontend_resources::FrameStatistics>().rendered_frames_count;
    if (current_frame == lastDrawnFrame)
        return;
    lastDrawnFrame = current_frame;

    std::string table_name = "##table";
    table_name += this->Name();

    ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable |
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
                            ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY;


    if (ImGui::Begin(this->Name())) {
        ImGui::BeginTabBar("#Lists");
        for (auto l = 0; l < p_in->GetParticleListCount(); ++l) {
            std::string label = "list ";
            label += std::to_string(l);
            if (ImGui::BeginTabItem(label.c_str())) {
                const auto& list = p_in->AccessParticles(l);
                int32_t num_columns = 0;
                ImGui::Text("particles: %lu", list.GetCount());
                ImGui::Text("global color: %u, %u, %u, %u", list.GetGlobalColour()[0], list.GetGlobalColour()[1],
                    list.GetGlobalColour()[2], list.GetGlobalColour()[3]);
                ImGui::Text("global radius: %f", list.GetGlobalRadius());
                ImGui::Text("vertex format: %s",
                    geocalls::SimpleSphericalParticles::VertexDataTypeNames[list.GetVertexDataType()].c_str());
                const auto v_comp =
                    geocalls::SimpleSphericalParticles::VertexDataTypeComponents[list.GetVertexDataType()];
                num_columns += v_comp;
                ImGui::Text("color format: %s",
                    geocalls::SimpleSphericalParticles::ColourDataTypeNames[list.GetColourDataType()].c_str());
                const auto c_comp =
                    geocalls::SimpleSphericalParticles::ColorDataTypeComponents[list.GetColourDataType()];
                num_columns += c_comp;
                ImGui::Text("direction format: %s",
                    geocalls::SimpleSphericalParticles::DirDataTypeNames[list.GetDirDataType()].c_str());
                const auto d_comp = geocalls::SimpleSphericalParticles::DirDataTypeComponents[list.GetDirDataType()];
                num_columns += d_comp;
                ImGui::Text(
                    "ID format: %s", geocalls::SimpleSphericalParticles::IDDataTypeNames[list.GetIDDataType()].c_str());
                const auto id_comp = geocalls::SimpleSphericalParticles::IDDataTypeComponents[list.GetIDDataType()];
                num_columns += id_comp;

                if (ImGui::BeginTable(table_name.c_str(), num_columns + 1, flags)) {
                    ImGui::TableSetupColumn("row", ImGuiTableColumnFlags_WidthFixed);
                    if (v_comp > 2) {
                        ImGui::TableSetupColumn("x", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("y", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("z", ImGuiTableColumnFlags_WidthFixed);
                        if (v_comp > 3) {
                            ImGui::TableSetupColumn("rad", ImGuiTableColumnFlags_WidthFixed);
                        }
                    }
                    if (c_comp == 1) {
                        ImGui::TableSetupColumn("i", ImGuiTableColumnFlags_WidthFixed);
                    } else if (c_comp > 2) {
                        ImGui::TableSetupColumn("r", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("g", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("b", ImGuiTableColumnFlags_WidthFixed);
                        if (c_comp > 3) {
                            ImGui::TableSetupColumn("a", ImGuiTableColumnFlags_WidthFixed);
                        }
                    }
                    if (d_comp > 0) {
                        ImGui::TableSetupColumn("vx", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("vy", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("vz", ImGuiTableColumnFlags_WidthFixed);
                    }
                    if (id_comp > 0) {
                        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed);
                    }

                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableHeadersRow();
                    ImGuiListClipper clipper;
                    const auto& storage = list.GetParticleStore();
                    const int modified_count =
                        static_cast<int>(std::min<UINT64>(list.GetCount(), std::numeric_limits<int>::max()));
                    clipper.Begin(modified_count);
                    while (clipper.Step()) {
                        for (auto row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                            ImGui::PushID(row);
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%010d", row);

                            if (v_comp > 2) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetXAcc()->Get_f(row));
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetYAcc()->Get_f(row));
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetZAcc()->Get_f(row));
                                if (v_comp > 3) {
                                    ImGui::TableNextColumn();
                                    ImGui::Text("%04.4f", storage.GetRAcc()->Get_f(row));
                                }
                            }
                            if (c_comp == 1) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetCRAcc()->Get_f(row));
                            } else if (c_comp > 2) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetCRAcc()->Get_f(row));
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetCGAcc()->Get_f(row));
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetCBAcc()->Get_f(row));
                                if (c_comp > 3) {
                                    ImGui::TableNextColumn();
                                    ImGui::Text("%04.4f", storage.GetCAAcc()->Get_f(row));
                                }
                            }
                            if (d_comp > 0) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetDXAcc()->Get_f(row));
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetDYAcc()->Get_f(row));
                                ImGui::TableNextColumn();
                                ImGui::Text("%04.4f", storage.GetDZAcc()->Get_f(row));
                            }
                            if (id_comp > 0) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%010d", storage.GetIDAcc()->Get_u64(row));
                            }

                            ImGui::PopID();
                        }
                    }
                    ImGui::EndTable();
                }
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}
