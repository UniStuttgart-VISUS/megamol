#include "SphereWidget.h"

#include "mmcore/param/IntParam.h"


megamol::thermodyn::SphereWidget::SphereWidget()
        : in_data_slot_("dataIn", "")
        , in_temp_slot_("tempIn", "")
        , in_dens_slot_("densIn", "")
        , flags_read_slot_("flagsRead", "")
        , particlelist_slot_("particlelist", "") {
    in_data_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_data_slot_);

    in_temp_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_temp_slot_);

    in_dens_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&in_dens_slot_);

    flags_read_slot_.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&flags_read_slot_);

    particlelist_slot_ << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&particlelist_slot_);
}


megamol::thermodyn::SphereWidget::~SphereWidget() {
    this->Release();
}


bool megamol::thermodyn::SphereWidget::create() {
    ctx_ = ImPlot::CreateContext();

    return true;
}


void megamol::thermodyn::SphereWidget::release() {
    ImPlot::DestroyContext(ctx_);
}


bool megamol::thermodyn::SphereWidget::Render(core::view::CallRender3DGL& call) {
    auto parts_in = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto temp_in = in_temp_slot_.CallAs<core::moldyn::MultiParticleDataCall>();

    auto dens_in = in_dens_slot_.CallAs<core::moldyn::MultiParticleDataCall>();

    parts_in->SetFrameID(call.Time());
    if (!(*parts_in)(0))
        return false;

    if (temp_in) {
        temp_in->SetFrameID(call.Time());
        if (!(*temp_in)(0))
            return false;
    }

    if (dens_in) {
        dens_in->SetFrameID(call.Time());
        if (!(*dens_in)(0))
            return false;
    }

    auto flags_read = flags_read_slot_.CallAs<core::FlagCallRead_CPU>();
    if (flags_read == nullptr)
        return false;

    if (!(*flags_read)(0))
        return false;

    parse_data(*parts_in, temp_in, dens_in, *flags_read);

    return true;
}


bool megamol::thermodyn::SphereWidget::GetExtents(core::view::CallRender3DGL& call) {
    auto parts_in = in_data_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (parts_in == nullptr)
        return false;

    auto temp_in = in_temp_slot_.CallAs<core::moldyn::MultiParticleDataCall>();

    auto dens_in = in_dens_slot_.CallAs<core::moldyn::MultiParticleDataCall>();

    parts_in->SetFrameID(call.Time());
    if (!(*parts_in)(1))
        return false;

    if (temp_in) {
        temp_in->SetFrameID(call.Time());
        if (!(*temp_in)(1))
            return false;
    }

    if (dens_in) {
        dens_in->SetFrameID(call.Time());
        if (!(*dens_in)(1))
            return false;
    }

    call.SetTimeFramesCount(parts_in->FrameCount());

    return true;
}


bool megamol::thermodyn::SphereWidget::widget(float x, float y, std::size_t idx,
    core::moldyn::SimpleSphericalParticles const& parts, core::moldyn::SimpleSphericalParticles const* temps,
    core::moldyn::SimpleSphericalParticles const* dens) {
    ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Appearing);

    // ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Appearing);

    bool plot_open = true;
    ImGui::Begin((std::string("Test Plot ") + std::to_string(idx)).c_str(), &plot_open,
        ImGuiWindowFlags_NoTitleBar); // | ImGuiWindowFlags_NoBackground);

    /*if (ImPlot::BeginPlot("data", nullptr, nullptr, ImVec2(-1, 0), ImPlotFlags_Query)) {
        ImPlot::EndPlot();
    }*/

    auto const id_acc = parts.GetParticleStore().GetIDAcc();
    auto const ic_acc = parts.GetParticleStore().GetCRAcc();

    ImGui::Text("Particle ID %d", id_acc->Get_u64(idx));
    // ImGui::Text("ICol Val %f", ic_acc->Get_f(idx));
    if (temps) {
        auto const temp_acc = temps->GetParticleStore().GetCRAcc();
        ImGui::Text("Temperature %f", temp_acc->Get_f(idx));
    }
    if (dens) {
        auto const dens_acc = dens->GetParticleStore().GetCRAcc();
        ImGui::Text("Density %f", dens_acc->Get_f(idx));
    }

    ImGui::End();

    return true;
}


bool megamol::thermodyn::SphereWidget::parse_data(core::moldyn::MultiParticleDataCall& in_parts,
    core::moldyn::MultiParticleDataCall* in_temps, core::moldyn::MultiParticleDataCall* in_dens,
    core::FlagCallRead_CPU& fcr) {
    auto const pl_count = in_parts.GetParticleListCount();

    if (pl_count == 0)
        return false;

    auto const selected_pl = particlelist_slot_.Param<core::param::IntParam>()->Value();

    if (selected_pl >= pl_count)
        return false;

    auto const& parts = in_parts.AccessParticles(selected_pl);

    auto const selection_data = fcr.getData();

    core::moldyn::SimpleSphericalParticles const* temps = nullptr;
    core::moldyn::SimpleSphericalParticles const* dens = nullptr;

    if (in_temps) {
        temps = &(in_temps->AccessParticles(selected_pl));
    }

    if (in_dens) {
        dens = &(in_dens->AccessParticles(selected_pl));
    }

    for (decltype(selection_data->flags)::element_type::size_type i = 0; i < selection_data->flags->size(); ++i) {
        auto const el = (*selection_data->flags)[i];
        if (el == core::FlagStorage::SELECTED) {
            widget(mouse_x_, mouse_y_, i, parts, temps, dens);
        }
    }

    return true;
}


bool megamol::thermodyn::SphereWidget::OnMouseMove(double x, double y) {
    mouse_x_ = static_cast<float>(x);
    mouse_y_ = static_cast<float>(y);
    return false;
}
