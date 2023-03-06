/*
 * Call.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "Call.h"
#include "InterfaceSlot.h"
#include "Module.h"
#include "widgets/ColorPalettes.h"

#ifdef MEGAMOL_USE_PROFILING
#include "ProfilingUtils.h"
#include "implot.h"
#define CALL_PROFILING_PLOT_HEIGHT (150.0f * megamol::gui::gui_scaling.Get())
#define CALL_PROFILING_WINDOW_WIDTH (300.0f * megamol::gui::gui_scaling.Get())
#endif // MEGAMOL_USE_PROFILING

using namespace megamol;
using namespace megamol::gui;


megamol::gui::Call::Call(ImGuiID uid, const std::string& class_name, const std::string& description,
    const std::string& plugin_name, const std::vector<std::string>& functions)
        : uid(uid)
        , class_name(class_name)
        , description(description)
        , plugin_name(plugin_name)
        , functions(functions)
        , connected_callslots()
        , caller_slot_name()
        , callee_slot_name()
        , gui_tooltip()
        , gui_profiling_button()
        , gui_profiling_btn_hovered(false)
        , gui_hidden(false)
        , gui_selected(false)
        , gui_set_active(false)
        , gui_set_hovered(false)
        , gui_hovered(false)
#ifdef MEGAMOL_USE_PROFILING
        , cpu_perf_history()
        , gl_perf_history()
        , profiling_parent_pointer(nullptr)
        , profiling_window_height(1.0f)
        , show_profiling_data(false)
        , gui_profiling_run_button()
        , pause_profiling_history_update(false)
        , profiling_button_position()
#endif // MEGAMOL_USE_PROFILING
{

    this->connected_callslots.emplace(CallSlotType::CALLER, nullptr);
    this->connected_callslots.emplace(CallSlotType::CALLEE, nullptr);
}

megamol::gui::Call::~Call() {

    // Disconnect call slots
    this->DisconnectCallSlots();
}


bool megamol::gui::Call::IsConnected() {

    unsigned int connected = 0;
    for (auto& callslot_map : this->connected_callslots) {
        if (callslot_map.second != nullptr) {
            connected++;
        }
    }
    if (connected != 2) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Call has only one connected call slot. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return (connected == 2);
}


bool megamol::gui::Call::ConnectCallSlots(
    megamol::gui::CallSlotPtr_t callslot_1, megamol::gui::CallSlotPtr_t callslot_2) {

    if ((callslot_1 == nullptr) || (callslot_2 == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if ((this->connected_callslots[callslot_1->Type()] != nullptr) ||
        (this->connected_callslots[callslot_2->Type()] != nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Call is already connected. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (callslot_1->IsConnectionValid((*callslot_2))) {
        this->connected_callslots[callslot_1->Type()] = callslot_1;
        this->connected_callslots[callslot_2->Type()] = callslot_2;
        return true;
    }
    return false;
}


bool megamol::gui::Call::DisconnectCallSlots(ImGuiID calling_callslot_uid) {

    try {
        for (auto& callslot_map : this->connected_callslots) {
            if (callslot_map.second != nullptr) {
                if (callslot_map.second->UID() != calling_callslot_uid) {
                    callslot_map.second->DisconnectCall(this->uid);
                }
                callslot_map.second.reset();
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


const megamol::gui::CallSlotPtr_t& megamol::gui::Call::CallSlotPtr(megamol::gui::CallSlotType type) {

    if (this->connected_callslots[type] == nullptr) {
        /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Returned pointer to call slot is nullptr. [%s,
        /// %s, line %d]\n",
        /// __FILE__, __FUNCTION__, __LINE__);
    }
    return this->connected_callslots[type];
}


void megamol::gui::Call::Draw(megamol::gui::PresentPhase phase, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        if (this->IsConnected()) {
            auto callerslot_ptr = this->CallSlotPtr(CallSlotType::CALLER);
            auto calleeslot_ptr = this->CallSlotPtr(CallSlotType::CALLEE);
            if ((callerslot_ptr == nullptr) || (calleeslot_ptr == nullptr)) {
                return;
            }
            this->caller_slot_name = callerslot_ptr->Name();
            this->callee_slot_name = calleeslot_ptr->Name();

            this->gui_hidden = false;
            bool connect_interface_slot = true;
            size_t curve_color_index = 0;
            if (callerslot_ptr->IsParentModuleConnected() && calleeslot_ptr->IsParentModuleConnected()) {

                // Calls lie only completely inside or outside groups
                if (callerslot_ptr->GetParentModule()->GroupUID() == calleeslot_ptr->GetParentModule()->GroupUID()) {
                    connect_interface_slot = false;
                    this->gui_hidden = callerslot_ptr->GetParentModule()->IsHidden();
                }

                if (state.interact.call_coloring_mode == 0) {
                    // Get curve color index depending on callee slot index
                    for (auto cs_ptr :
                        calleeslot_ptr->GetParentModule()->CallSlots(megamol::gui::CallSlotType::CALLEE)) {
                        if (cs_ptr->UID() != calleeslot_ptr->UID()) {
                            curve_color_index++;
                        } else {
                            break;
                        }
                    }
                } else if (state.interact.call_coloring_mode == 1) {
                    // Get curve color index depending on calling module
                    curve_color_index = callerslot_ptr->GetParentModule()->UID();
                }
            }

            if (!this->gui_hidden) {

                ImVec2 caller_pos = callerslot_ptr->Position();
                ImVec2 callee_pos = calleeslot_ptr->Position();
                if (connect_interface_slot) {
                    if (callerslot_ptr->InterfaceSlotPtr() != nullptr) {
                        caller_pos = callerslot_ptr->InterfaceSlotPtr()->Position();
                    }
                    if (calleeslot_ptr->InterfaceSlotPtr() != nullptr) {
                        callee_pos = calleeslot_ptr->InterfaceSlotPtr()->Position();
                    }
                }

                ImGui::PushID(static_cast<int>(this->uid));

                /// COLOR_CALL_BACKGROUND
                ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_CALL_HIGHTLIGHT
                tmpcol = style.Colors[ImGuiCol_FrameBgActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_CALL_CURVE
                tmpcol = style.Colors[ImGuiCol_FrameBgHovered];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                /// See ColorPalettes.h for all predefined color palettes:
                if (state.interact.call_coloring_map == 1) {
                    // Set3Map(12):
                    const size_t map_size = 12;
                    tmpcol = ImVec4(Set3Map[(curve_color_index % map_size)][0],
                        Set3Map[(curve_color_index % map_size)][1], Set3Map[(curve_color_index % map_size)][2], 1.0f);
                } else if (state.interact.call_coloring_map == 2) {
                    // PairedMap(12):
                    const size_t map_size = 12;
                    tmpcol = ImVec4(PairedMap[(curve_color_index % map_size)][0],
                        PairedMap[(curve_color_index % map_size)][1], PairedMap[(curve_color_index % map_size)][2],
                        1.0f);
                }
                const ImU32 COLOR_CALL_CURVE = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_CALL_CURVE_HIGHLIGHT
                tmpcol = style.Colors[ImGuiCol_ButtonActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_CURVE_HIGHLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_CALL_GROUP_BORDER
                tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_CALL_GROUP_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_TEXT
                const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);

                auto bez_p1 = caller_pos;
                auto bez_p2 = caller_pos + ImVec2((50.0f * megamol::gui::gui_scaling.Get()), 0.0f);
                auto bez_p3 = callee_pos + ImVec2((-50.0f * megamol::gui::gui_scaling.Get()), 0.0f);
                auto bez_p4 = callee_pos;
                auto bez_linewidth = GUI_LINE_THICKNESS * state.canvas.zooming;

                if (state.interact.call_show_label || state.interact.call_show_slots_label) {
                    std::string slots_label = this->SlotsLabel();
                    auto slots_label_width = ImGui::CalcTextSize(slots_label.c_str()).x;
                    auto class_name_width = ImGui::CalcTextSize(this->class_name.c_str()).x;
                    ImVec2 call_center = ImVec2(caller_pos.x + (callee_pos.x - caller_pos.x) / 2.0f,
                        caller_pos.y + (callee_pos.y - caller_pos.y) / 2.0f);
                    auto call_name_width = 0.0f;
                    if (state.interact.call_show_label) {
                        call_name_width = std::max(call_name_width, class_name_width);
                    }
                    if (state.interact.call_show_slots_label) {
                        call_name_width = std::max(call_name_width, slots_label_width);
                    }
                    ImVec2 rect_size = ImVec2(call_name_width + (2.0f * style.ItemSpacing.x * state.canvas.zooming),
                        ImGui::GetFontSize() + (2.0f * style.ItemSpacing.y * state.canvas.zooming));
                    if (state.interact.call_show_label && state.interact.call_show_slots_label) {
                        rect_size.y += (ImGui::GetFontSize() + (style.ItemSpacing.y * state.canvas.zooming));
                    }
#ifdef MEGAMOL_USE_PROFILING
                    rect_size.x += 2.0f * ImGui::GetTextLineHeightWithSpacing();
                    rect_size.y = std::max(rect_size.y, ImGui::GetTextLineHeightWithSpacing());
#endif // MEGAMOL_USE_PROFILING
                    ImVec2 call_rect_min =
                        ImVec2(call_center.x - (rect_size.x / 2.0f), call_center.y - (rect_size.y / 2.0f));
#ifdef MEGAMOL_USE_PROFILING
                    /*
                    /// Draw profiling data inplace
                    if (this->show_profiling_data) {
                        rect_size = ImVec2(((CALL_PROFILING_WINDOW_WIDTH * state.canvas.zooming) +
                                               (style.ItemSpacing.x * 2.0f * state.canvas.zooming)),
                            ((this->profiling_window_height * state.canvas.zooming) +
                                (style.ItemSpacing.y * 2.0f * state.canvas.zooming) + rect_size.y));
                    }
                    */
#endif // MEGAMOL_USE_PROFILING
                    ImVec2 call_rect_max = ImVec2((call_rect_min.x + rect_size.x), (call_rect_min.y + rect_size.y));

                    const float min_curve_zoom = 0.2f;
                    const std::string button_label = "call_" + std::to_string(this->uid);

                    if (phase == megamol::gui::PresentPhase::INTERACTION) {

                        // Button
                        ImGui::SetCursorScreenPos(call_rect_min);
                        ImGui::SetItemAllowOverlap();
                        ImGui::InvisibleButton(button_label.c_str(), rect_size, ImGuiButtonFlags_NoSetKeyOwner);
                        ImGui::SetItemAllowOverlap();

                        /// Draw simple line if zooming is too small for nice bezier curves.
                        auto mouse_pos = ImGui::GetMousePos();
                        ImVec2 diff_vec = mouse_pos;
                        if (state.canvas.zooming < min_curve_zoom) {
                            diff_vec -= ImLineClosestPoint(bez_p1, bez_p4, mouse_pos);
                        } else {
                            diff_vec -= ImBezierCubicClosestPoint(bez_p1, bez_p2, bez_p3, bez_p4, mouse_pos, 10.0f);
                        }
                        auto curve_hovered = (fabs(diff_vec.x) <= std::max(1.0f, bez_linewidth / 2.0f)) &&
                                             (fabs(diff_vec.y) <= std::max(1.0f, bez_linewidth / 2.0f));

                        if (this->gui_set_active || ImGui::IsItemActivated() ||
                            (curve_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left))) {
                            state.interact.button_active_uid = this->uid;
                            this->gui_set_active = false;
                        }
                        if (this->gui_set_hovered || ImGui::IsItemHovered() || curve_hovered) {
                            state.interact.button_hovered_uid = this->uid;
                            this->gui_set_hovered = false;
                        }

                        // Context Menu
                        ImGui::PushFont(state.canvas.gui_font_ptr);
                        const std::string call_context_menu = "call_context_menu";
                        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right) &&
                            (ImGui::IsItemHovered() || curve_hovered)) {
                            ImGui::OpenPopup(call_context_menu.c_str());
                        }
                        if (ImGui::BeginPopup(call_context_menu.c_str())) {
                            state.interact.button_active_uid = this->uid;

                            ImGui::TextDisabled("Call");
                            ImGui::Separator();

                            if (ImGui::MenuItem("Delete",
                                    state.hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM].keycode.ToString().c_str())) {
                                state.interact.process_deletion = true;
                            }
                            ImGui::Separator();

                            ImGui::TextDisabled("Description");
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                            ImGui::TextUnformatted(this->description.c_str());
                            ImGui::PopTextWrapPos();
                            ImGui::EndPopup();
                        }

                        // Hover Tooltip
                        if (!state.interact.call_show_slots_label) {
                            if (state.interact.call_hovered_uid == this->uid && !this->gui_profiling_btn_hovered) {
                                this->gui_tooltip.ToolTip(slots_label, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);

                            } else {
                                this->gui_tooltip.Reset();
                            }
                        }
                        ImGui::PopFont();

                    } else if (phase == megamol::gui::PresentPhase::RENDERING) {

                        bool gui_active = (state.interact.button_active_uid == this->uid);
                        this->gui_hovered = (state.interact.button_hovered_uid == this->uid);
                        bool mouse_clicked_anywhere =
                            ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiPopupFlags_MouseButtonLeft);

                        // Draw Curve
                        ImU32 color_curve = COLOR_CALL_CURVE;
                        if (this->gui_hovered || this->gui_selected) {
                            color_curve = COLOR_CALL_CURVE_HIGHLIGHT;
                        }
                        /// Draw simple line if zooming is too small for nice bezier curves.
                        if (state.canvas.zooming < min_curve_zoom) {
                            draw_list->AddLine(caller_pos, callee_pos, color_curve, bez_linewidth);
                        } else {
                            draw_list->AddBezierCubic(bez_p1, bez_p2, bez_p3, bez_p4, color_curve, bez_linewidth, 0);
                        }

                        // Selection
                        if (!this->gui_selected && gui_active) {
                            state.interact.call_selected_uid = this->uid;
                            this->gui_selected = true;
                            state.interact.callslot_selected_uid = GUI_INVALID_ID;
                            state.interact.modules_selected_uids.clear();
                            state.interact.group_selected_uid = GUI_INVALID_ID;
                            state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                        }
                        // Deselection
                        else if (this->gui_selected && ((mouse_clicked_anywhere && !this->gui_hovered) ||
                                                           (state.interact.call_selected_uid != this->uid))) {
                            this->gui_selected = false;
                            if (state.interact.call_selected_uid == this->uid) {
                                state.interact.call_selected_uid = GUI_INVALID_ID;
                            }
                        }

                        // Hovering
                        if (this->gui_hovered) {
                            state.interact.call_hovered_uid = this->uid;
                        }
                        if (!this->gui_hovered && (state.interact.call_hovered_uid == this->uid)) {
                            state.interact.call_hovered_uid = GUI_INVALID_ID;
                        }

                        // Draw Background
                        ImU32 call_bg_color = (this->gui_selected || this->gui_hovered) ? (COLOR_CALL_HIGHTLIGHT)
                                                                                        : (COLOR_CALL_BACKGROUND);
                        draw_list->AddRectFilled(call_rect_min, call_rect_max, call_bg_color, GUI_RECT_CORNER_RADIUS);
                        draw_list->AddRect(
                            call_rect_min, call_rect_max, COLOR_CALL_GROUP_BORDER, GUI_RECT_CORNER_RADIUS);

#ifdef MEGAMOL_USE_PROFILING
                        // Lazy loading of performance button texture
                        const auto profiling_button_size = ImGui::GetTextLineHeight();
                        if (!this->gui_profiling_button.IsLoaded()) {
                            this->gui_profiling_button.LoadTextureFromFile(GUI_FILENAME_TEXTURE_PROFILING_BUTTON);
                        }
                        ImVec2 profiling_button_pos =
                            ImVec2(call_rect_min.x + (style.ItemInnerSpacing.x * state.canvas.zooming),
                                call_center.y - (profiling_button_size / 2.0f));
                        ImGui::SetCursorScreenPos(profiling_button_pos);
                        ImGui::PushFont(state.canvas.gui_font_ptr);
                        this->profiling_button_position =
                            ImVec2(ImGui::GetCursorScreenPos().x + profiling_button_size / 2.0f, call_rect_max.y);
                        if (this->gui_profiling_button.Button(
                                "Profiling", ImVec2(profiling_button_size, profiling_button_size))) {
                            this->show_profiling_data = !this->show_profiling_data;
                            if (this->show_profiling_data) {
                                state.interact.profiling_show = true;
                            }
                        }
                        this->gui_profiling_btn_hovered = ImGui::IsItemHovered();
                        ImGui::PopFont();
                        if (this->show_profiling_data) {
                            this->pause_profiling_history_update = state.interact.profiling_pause_update;
                            /*
                            /// Draw profiling data inplace
                            ImGui::SetCursorScreenPos(
                                ImVec2(call_rect_min.x + (style.ItemSpacing.x * state.canvas.zooming),
                                    call_center.y + (call_center.y - call_rect_min.y)));
                            ImGui::BeginChild("call_profiling_info",
                                ImVec2((CALL_PROFILING_WINDOW_WIDTH * state.canvas.zooming),
                                    (this->profiling_window_height * state.canvas.zooming)),
                                true,
                                ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_NoMove |
                                    ImGuiWindowFlags_NoScrollbar);
                            ImGui::TextUnformatted("Profiling");
                            ImGui::SameLine();
                            // Lazy loading of run button textures
                            if (!this->gui_profiling_run_button.IsLoaded()) {
                                this->gui_profiling_run_button.LoadTextureFromFile(
                                    GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PAUSE,
                                    GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PLAY);
                            }
                            this->gui_profiling_run_button.ToggleButton(state.interact.profiling_pause_update,
                                "Pause update of profiling values globally", "Continue updating of profiling values",
                                ImVec2(ImGui::GetTextLineHeight(), ImGui::GetTextLineHeight()));
                            ImGui::TextDisabled("Callback Name:");
                            this->DrawProfiling(state);
                            ImGui::EndChild();
                            */
                        }
#endif // MEGAMOL_USE_PROFILING
                        // Draw Text
                        ImVec2 text_pos_left_upper =
                            (call_center + ImVec2(-(class_name_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                        if (state.interact.call_show_label && state.interact.call_show_slots_label) {
                            text_pos_left_upper.y -= (0.5f * ImGui::GetFontSize());
                        }
                        if (state.interact.call_show_label) {
                            draw_list->AddText(text_pos_left_upper,
                                ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]), this->class_name.c_str());
                        }

                        text_pos_left_upper =
                            (call_center + ImVec2(-(slots_label_width / 2.0f), -0.5f * ImGui::GetFontSize()));
                        if (state.interact.call_show_label && state.interact.call_show_slots_label) {
                            text_pos_left_upper.y += (0.5f * ImGui::GetFontSize());
                        }
                        if (state.interact.call_show_slots_label) {
                            // Caller
                            draw_list->AddText(text_pos_left_upper,
                                ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLER), this->caller_slot_name.c_str());
                            // Separator
                            text_pos_left_upper.x += ImGui::CalcTextSize(this->caller_slot_name.c_str()).x;
                            draw_list->AddText(text_pos_left_upper, COLOR_TEXT, this->slot_name_separator.c_str());
                            // Callee
                            text_pos_left_upper.x += ImGui::CalcTextSize(this->slot_name_separator.c_str()).x;
                            draw_list->AddText(text_pos_left_upper,
                                ImGui::ColorConvertFloat4ToU32(GUI_COLOR_SLOT_CALLEE), this->callee_slot_name.c_str());
                        }
                    }
                }

                ImGui::PopID();
            }
        }
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


#ifdef MEGAMOL_USE_PROFILING

void megamol::gui::Call::AppendPerformanceData(frontend_resources::PerformanceManager::frame_type frame,
    const frontend_resources::PerformanceManager::timer_entry& entry) {
    if (!this->pause_profiling_history_update) {
        switch (entry.api) {
        case frontend_resources::PerformanceManager::query_api::CPU:
            this->cpu_perf_history[entry.user_index].push_sample(frame, entry.frame_index,
                std::chrono::duration<double, std::milli>(entry.timestamp.time_since_epoch()).count());
            break;
        case frontend_resources::PerformanceManager::query_api::OPENGL:
            this->gl_perf_history[entry.user_index].push_sample(frame, entry.frame_index,
                std::chrono::duration<double, std::milli>(entry.timestamp.time_since_epoch()).count());
            break;
        }
    }
}


void megamol::gui::Call::DrawProfiling(GraphItemsState_t& state) {

    ImGui::BeginTabBar("profiling", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyScroll);
    auto func_cnt = this->cpu_perf_history.size();
    for (size_t i = 0; i < func_cnt; i++) {
        auto& tab_label = this->cpu_perf_history[i].get_name(); // this->profiling[i].name;
        if (ImGui::BeginTabItem(tab_label.c_str(), nullptr, ImGuiTabItemFlags_None)) {

            static ProfilingUtils::MetricType display_idx = ProfilingUtils::MetricType::MINMAXAVG;
            static ImPlotAxisFlags y_flags = ImPlotAxisFlags_AutoFit;
            ProfilingUtils::MetricDropDown(display_idx, y_flags);

            if (ImGui::BeginTable(("table_" + tab_label).c_str(), 2,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize,
                    ImVec2(0.0f, 0.0f))) {
                ImGui::TableSetupColumn(("column_" + tab_label).c_str(), ImGuiTableColumnFlags_WidthStretch);
                ProfilingUtils::PrintTableRow("Min CPU Time",
                    this->cpu_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::MIN,
                        core::MultiPerformanceHistory::metric_type::MIN));
                ProfilingUtils::PrintTableRow("Average CPU Time",
                    this->cpu_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::AVERAGE,
                        core::MultiPerformanceHistory::metric_type::AVERAGE));
                ProfilingUtils::PrintTableRow("Max CPU Time",
                    this->cpu_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::MAX,
                        core::MultiPerformanceHistory::metric_type::MAX));
                ProfilingUtils::PrintTableRow("Max CPU Samples / Frame",
                    static_cast<int>(
                        this->cpu_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::MAX,
                            core::MultiPerformanceHistory::metric_type::COUNT)));
                ProfilingUtils::PrintTableRow("Num CPU Samples", static_cast<int>(this->cpu_perf_history[i].samples()));
                ImGui::EndTable();
            }

            ProfilingUtils::DrawPlot("CPU History",
                ImVec2(ImGui::GetContentRegionAvail().x, (CALL_PROFILING_PLOT_HEIGHT)), y_flags, display_idx,
                cpu_perf_history[i]);

            if (ImGui::BeginTable(("table_" + tab_label).c_str(), 2,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize,
                    ImVec2(0.0f, 0.0f))) {
                ImGui::TableSetupColumn(("column_" + tab_label).c_str(), ImGuiTableColumnFlags_WidthStretch);
                ProfilingUtils::PrintTableRow("Min GL Time",
                    this->gl_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::MIN,
                        core::MultiPerformanceHistory::metric_type::MIN));
                ProfilingUtils::PrintTableRow("Average GL Time",
                    this->gl_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::AVERAGE,
                        core::MultiPerformanceHistory::metric_type::AVERAGE));
                ProfilingUtils::PrintTableRow("Max GL Time",
                    this->gl_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::MAX,
                        core::MultiPerformanceHistory::metric_type::MAX));
                ProfilingUtils::PrintTableRow("Max GL Samples / Frame",
                    static_cast<int>(
                        this->gl_perf_history[i].window_statistics(core::MultiPerformanceHistory::metric_type::MAX,
                            core::MultiPerformanceHistory::metric_type::COUNT)));
                ProfilingUtils::PrintTableRow("Num GL Samples", static_cast<int>(this->gl_perf_history[i].samples()));
                ImGui::EndTable();
            }

            ProfilingUtils::DrawPlot("GL History",
                ImVec2(ImGui::GetContentRegionAvail().x, (CALL_PROFILING_PLOT_HEIGHT)), y_flags, display_idx,
                gl_perf_history[i]);

            ImGui::EndTabItem();
            y_flags = 0;
        }
    }
    ImGui::EndTabBar();

    this->profiling_window_height = std::max(1.0f, ImGui::GetCursorPosY());
}


#endif // MEGAMOL_USE_PROFILING
