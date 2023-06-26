/*
 * Module.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "Module.h"
#include "Call.h"
#include "CallSlot.h"
#include "InterfaceSlot.h"

#ifdef MEGAMOL_USE_PROFILING
#include "ProfilingUtils.h"
#include "implot.h"
#define MODULE_PROFILING_PLOT_HEIGHT (150.0f * megamol::gui::gui_scaling.Get())
#define MODULE_PROFILING_WINDOW_WIDTH (300.0f * megamol::gui::gui_scaling.Get())
#endif // MEGAMOL_USE_PROFILING

using namespace megamol;
using namespace megamol::gui;


megamol::gui::Module::Module(ImGuiID uid, const std::string& class_name, const std::string& description,
    const std::string& plugin_name, bool is_view)
        : uid(uid)
        , class_name(class_name)
        , description(description)
        , plugin_name(plugin_name)
        , is_view(is_view)
        , parameters()
        , callslots()
        , name("")
        , graph_entry_name("")
        , group_uid(GUI_INVALID_ID)
        , group_name("")
        , gui_param_groups()
        , gui_position(ImVec2(FLT_MAX, FLT_MAX))
        , gui_size(ImVec2(0.0f, 0.0f))
        , gui_update(true)
        , gui_param_child_show(false)
        , gui_set_screen_position(ImVec2(FLT_MAX, FLT_MAX))
        , gui_set_selected_slot_position(false)
        , gui_hidden(false)
        , gui_other_item_hovered(false)
        , gui_tooltip()
        , gui_rename_popup()
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
        , gui_profiling_button()
        , gui_profiling_run_button()
        , pause_profiling_history_update(false)
        , profiling_button_position()
#endif // MEGAMOL_USE_PROFILING
{

    this->callslots.emplace(megamol::gui::CallSlotType::CALLER, CallSlotPtrVector_t());
    this->callslots.emplace(megamol::gui::CallSlotType::CALLEE, CallSlotPtrVector_t());
}


megamol::gui::Module::~Module() {

    // Delete all call slots
    this->DeleteCallSlots();
}


bool megamol::gui::Module::AddCallSlot(megamol::gui::CallSlotPtr_t callslot) {

    if (callslot == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to given call slot is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto type = callslot->Type();
    for (auto& callslot_ptr : this->callslots[type]) {
        if (callslot_ptr == callslot) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Pointer to call slot already registered in modules call slot list. [%s, %s, line %d]\n",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    }
    this->callslots[type].emplace_back(callslot);
    return true;
}


bool megamol::gui::Module::DeleteCallSlots() {

    try {
        for (auto& callslots_map : this->callslots) {
            for (auto callslot_iter = callslots_map.second.begin(); callslot_iter != callslots_map.second.end();
                 callslot_iter++) {
                (*callslot_iter)->DisconnectCalls();
                (*callslot_iter)->DisconnectParentModule();

                if ((*callslot_iter).use_count() > 1) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unclean deletion. Found %i references pointing to call slot. [%s, %s, line %d]\n",
                        (*callslot_iter).use_count(), __FILE__, __FUNCTION__, __LINE__);
                }

                (*callslot_iter).reset();
            }
            callslots_map.second.clear();
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


CallSlotPtr_t megamol::gui::Module::CallSlotPtr(ImGuiID callslot_uid) {

    if (callslot_uid != GUI_INVALID_ID) {
        for (auto& callslot_map : this->CallSlots()) {
            for (auto& callslot : callslot_map.second) {
                if (callslot->UID() == callslot_uid) {
                    return callslot;
                }
            }
        }
    }
    return nullptr;
}


void megamol::gui::Module::Draw(megamol::gui::PresentPhase phase, megamol::gui::GraphItemsState_t& state) {

    if (ImGui::GetCurrentContext() == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    try {
        // Update size
        if (this->gui_update || (this->gui_size.x <= 0.0f) || (this->gui_size.y <= 0.0f)) {
            this->update(state);
            this->gui_update = false;
        }

        // Init position of newly created module (check after size update)
        if (this->gui_set_screen_position != ImVec2(FLT_MAX, FLT_MAX)) {
            this->gui_position = (this->gui_set_screen_position - state.canvas.offset) / state.canvas.zooming;
            this->gui_set_screen_position = ImVec2(FLT_MAX, FLT_MAX);
        }
        // Init position using current compatible slot
        if (this->gui_set_selected_slot_position) {
            for (auto& callslot_map : this->CallSlots()) {
                for (auto& callslot_ptr : callslot_map.second) {
                    CallSlotType callslot_type = (callslot_ptr->Type() == CallSlotType::CALLEE)
                                                     ? (CallSlotType::CALLER)
                                                     : (CallSlotType::CALLEE);
                    for (auto& call_ptr : callslot_ptr->GetConnectedCalls()) {
                        auto connected_callslot_ptr = call_ptr->CallSlotPtr(callslot_type);
                        float call_width =
                            (4.0f * GUI_GRAPH_BORDER + ImGui::CalcTextSize(call_ptr->ClassName().c_str()).x);
                        if (state.interact.callslot_selected_uid != GUI_INVALID_ID) {
                            if ((connected_callslot_ptr->UID() == state.interact.callslot_selected_uid) &&
                                connected_callslot_ptr->IsParentModuleConnected()) {
                                ImVec2 module_size = connected_callslot_ptr->GetParentModule()->Size();
                                ImVec2 module_pos = connected_callslot_ptr->GetParentModule()->Position();
                                if (connected_callslot_ptr->Type() == CallSlotType::CALLEE) {
                                    this->gui_position = module_pos - ImVec2((call_width + this->gui_size.x), 0.0f);
                                } else {
                                    this->gui_position = module_pos + ImVec2((call_width + module_size.x), 0.0f);
                                }
                                break;
                            }
                        } else if ((state.interact.interfaceslot_selected_uid != GUI_INVALID_ID) &&
                                   (connected_callslot_ptr->InterfaceSlotPtr() != nullptr)) {
                            if (state.interact.interfaceslot_selected_uid ==
                                connected_callslot_ptr->InterfaceSlotPtr()->UID()) {
                                ImVec2 interfaceslot_position =
                                    (connected_callslot_ptr->InterfaceSlotPtr()->Position() - state.canvas.offset) /
                                    state.canvas.zooming;
                                if (connected_callslot_ptr->Type() == CallSlotType::CALLEE) {
                                    this->gui_position =
                                        interfaceslot_position - ImVec2((call_width + this->gui_size.x), 0.0f);
                                } else {
                                    this->gui_position = interfaceslot_position + ImVec2(call_width, 0.0f);
                                }
                                break;
                            }
                        }
                    }
                }
            }
            this->gui_set_selected_slot_position = false;
        }
        if ((this->gui_position.x == FLT_MAX) && (this->gui_position.y == FLT_MAX)) {
            // See layout border_offset in Graph::Presentation::layout_graph
            this->gui_position = megamol::gui::Module::GetDefaultModulePosition(state.canvas);
        }

        if (!this->gui_hidden) {
            bool mouse_clicked_anywhere =
                ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiPopupFlags_MouseButtonLeft);

            ImGui::PushID(static_cast<int>(this->uid));

            // Get current module information
            ImVec2 module_size = this->gui_size * state.canvas.zooming;
            ImVec2 module_rect_min = state.canvas.offset + (this->gui_position * state.canvas.zooming);
            ImVec2 module_rect_max = module_rect_min + module_size;
            ImVec2 module_center = module_rect_min + ImVec2(module_size.x / 2.0f, module_size.y / 2.0f);

            /// COLOR_MODULE_BACKGROUND
            ImVec4 tmpcol = style.Colors[ImGuiCol_FrameBg];
            tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
            const ImU32 COLOR_MODULE_BACKGROUND = ImGui::ColorConvertFloat4ToU32(tmpcol);

            // MODULE ------------------------------------------------------
            std::string button_label = "module_" + std::to_string(this->uid);

            if (phase == megamol::gui::PresentPhase::INTERACTION) {

                // Button
                ImGui::SetCursorScreenPos(module_rect_min);
                ImGui::SetItemAllowOverlap();
                ImGui::InvisibleButton(button_label.c_str(), module_size, ImGuiButtonFlags_NoSetKeyOwner);
                ImGui::SetItemAllowOverlap();
                if (this->gui_set_active || ImGui::IsItemActivated()) {
                    state.interact.button_active_uid = this->uid;
                    this->gui_set_active = false;
                }
                if (this->gui_set_hovered || ImGui::IsItemHovered()) {
                    state.interact.button_hovered_uid = this->uid;
                    this->gui_set_hovered = false;
                }

                ImGui::PushFont(state.canvas.gui_font_ptr);

                // Context menu
                bool popup_rename = false;
                if (ImGui::BeginPopupContextItem("invisible_button_context")) {
                    state.interact.button_active_uid = this->uid;
                    bool singleselect = ((state.interact.modules_selected_uids.size() == 1) &&
                                         (this->found_uid(state.interact.modules_selected_uids, this->uid)));

                    ImGui::TextDisabled("Module");
                    ImGui::Separator();

                    if (ImGui::MenuItem("Delete",
                            state.hotkeys[HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM].keycode.ToString().c_str())) {
                        state.interact.process_deletion = true;
                    }
                    if (ImGui::MenuItem("Layout Modules", nullptr, false, !singleselect)) {
                        state.interact.modules_layout = true;
                    }
                    if (ImGui::MenuItem("Rename", nullptr, false, singleselect)) {
                        popup_rename = true;
                    }
                    if (ImGui::BeginMenu("Add to Group", true)) {
                        if (ImGui::MenuItem("New")) {
                            state.interact.modules_add_group_uids.clear();
                            if (this->gui_selected) {
                                for (auto& module_uid : state.interact.modules_selected_uids) {
                                    state.interact.modules_add_group_uids.emplace_back(
                                        UIDPair_t(module_uid, GUI_INVALID_ID));
                                }
                            } else {
                                state.interact.modules_add_group_uids.emplace_back(
                                    UIDPair_t(this->uid, GUI_INVALID_ID));
                            }
                        }
                        if (!state.groups.empty()) {
                            ImGui::Separator();
                        }
                        for (auto& group_pair : state.groups) {
                            if (ImGui::MenuItem(group_pair.second.c_str())) {
                                state.interact.modules_add_group_uids.clear();
                                if (this->gui_selected) {
                                    for (auto& module_uid : state.interact.modules_selected_uids) {
                                        state.interact.modules_add_group_uids.emplace_back(
                                            UIDPair_t(module_uid, group_pair.first));
                                    }
                                } else {
                                    state.interact.modules_add_group_uids.emplace_back(
                                        UIDPair_t(this->uid, group_pair.first));
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }
                    if (ImGui::MenuItem("Remove from Group", nullptr, false, (this->group_uid != GUI_INVALID_ID))) {
                        state.interact.modules_remove_group_uids.clear();
                        if (this->gui_selected) {
                            for (auto& module_uid : state.interact.modules_selected_uids) {
                                state.interact.modules_remove_group_uids.emplace_back(module_uid);
                            }
                        } else {
                            state.interact.modules_remove_group_uids.emplace_back(this->uid);
                        }
                    }

                    if (singleselect) {
                        ImGui::Separator();
                        ImGui::TextDisabled("Description");
                        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 13.0f);
                        ImGui::TextUnformatted(this->description.c_str());
                        ImGui::PopTextWrapPos();
                    }
                    ImGui::EndPopup();
                }

                // Hover Tooltip
                if ((state.interact.module_hovered_uid == this->uid) && !state.interact.module_show_label &&
                    !this->gui_other_item_hovered) {
                    ImGui::PushFont(state.canvas.gui_font_ptr);
                    this->gui_tooltip.ToolTip(this->name, ImGui::GetID(button_label.c_str()), 0.5f, 5.0f);
                    ImGui::PopFont();
                } else {
                    this->gui_tooltip.Reset();
                }

                // Rename pop-up
                std::string new_name = this->name;
                std::string last_module_name = this->FullName();
                if (this->gui_rename_popup.Rename("Rename Module", popup_rename, new_name)) {
                    this->SetName(new_name);
                    this->Update();
                    if (state.interact.graph_is_running) {
                        state.interact.module_rename.push_back(StrPair_t(last_module_name, this->FullName()));
                    }
                }

                ImGui::PopFont();

            } else if (phase == megamol::gui::PresentPhase::RENDERING) {

                bool active = (state.interact.button_active_uid == this->uid);
                this->gui_hovered = (state.interact.button_hovered_uid == this->uid);

                // Selection
                if (!this->gui_selected &&
                    (active || this->found_uid(state.interact.modules_selected_uids, this->uid))) {
                    if (!this->found_uid(state.interact.modules_selected_uids, this->uid)) {
                        if (ImGui::IsKeyPressed(ImGuiMod_Shift)) {
                            // Multiple Selection
                            this->add_uid(state.interact.modules_selected_uids, this->uid);
                        } else {
                            // Single Selection
                            state.interact.modules_selected_uids.clear();
                            state.interact.modules_selected_uids.emplace_back(this->uid);
                        }
                    }
                    this->gui_selected = true;
                    state.interact.callslot_selected_uid = GUI_INVALID_ID;
                    state.interact.call_selected_uid = GUI_INVALID_ID;
                    state.interact.group_selected_uid = GUI_INVALID_ID;
                    state.interact.interfaceslot_selected_uid = GUI_INVALID_ID;
                }
                // Deselection
                else if (this->gui_selected &&
                         ((mouse_clicked_anywhere && (state.interact.module_hovered_uid == GUI_INVALID_ID) &&
                              !ImGui::IsKeyPressed(ImGuiMod_Shift)) ||
                             (active && ImGui::IsKeyPressed(ImGuiMod_Shift)) ||
                             (!this->found_uid(state.interact.modules_selected_uids, this->uid)))) {
                    this->gui_selected = false;
                    this->erase_uid(state.interact.modules_selected_uids, this->uid);
                }

                // Dragging
                if (this->gui_selected && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(0)) {
                    this->gui_position += (ImGui::GetIO().MouseDelta / state.canvas.zooming);
                    this->update(state);
                }

                // Hovering
                if (this->gui_hovered) {
                    state.interact.module_hovered_uid = this->uid;
                }
                if (!this->gui_hovered && (state.interact.module_hovered_uid == this->uid)) {
                    state.interact.module_hovered_uid = GUI_INVALID_ID;
                }

                /// COLOR_MODULE_HIGHTLIGHT
                tmpcol = style.Colors[ImGuiCol_FrameBgActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_MODULE_HIGHTLIGHT = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_MODULE_BORDER
                tmpcol = style.Colors[ImGuiCol_ScrollbarGrabActive];
                tmpcol = ImVec4(tmpcol.x * tmpcol.w, tmpcol.y * tmpcol.w, tmpcol.z * tmpcol.w, 1.0f);
                const ImU32 COLOR_MODULE_BORDER = ImGui::ColorConvertFloat4ToU32(tmpcol);
                /// COLOR_HEADER
                const ImU32 COLOR_HEADER = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBgHovered]);
                /// COLOR_HEADER_HIGHLIGHT
                const ImU32 COLOR_HEADER_HIGHLIGHT =
                    ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
                /// COLOR_TEXT
                const ImU32 COLOR_TEXT = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);

                // Draw Background
                ImU32 module_bg_color =
                    (this->gui_selected || this->gui_hovered) ? (COLOR_MODULE_HIGHTLIGHT) : (COLOR_MODULE_BACKGROUND);
                draw_list->AddRectFilled(module_rect_min, module_rect_max, module_bg_color, GUI_RECT_CORNER_RADIUS,
                    ImDrawFlags_RoundCornersAll);

                // Draw Text and Option Buttons
                float text_width;
                ImVec2 text_pos_left_upper;
                const float line_height = ImGui::GetTextLineHeightWithSpacing();
                bool graph_entry_button = this->is_view;
                bool parameter_button = (!this->parameters.empty());
                bool profiling_button = false;
#ifdef MEGAMOL_USE_PROFILING
                profiling_button = true;
#endif // MEGAMOL_USE_PROFILING
                unsigned int button_count = (graph_entry_button) ? (1) : (0);
                button_count += (parameter_button) ? (1) : (0);
                button_count += (profiling_button) ? (1) : (0);

                this->gui_other_item_hovered = false;
                if (state.interact.module_show_label) {

                    auto header_color =
                        (this->gui_selected || this->gui_hovered) ? (COLOR_HEADER_HIGHLIGHT) : (COLOR_HEADER);
                    ImVec2 header_rect_max =
                        module_rect_min + ImVec2(module_size.x, ImGui::GetTextLineHeightWithSpacing());
                    draw_list->AddRectFilled(module_rect_min, header_rect_max, header_color, GUI_RECT_CORNER_RADIUS,
                        (ImDrawFlags_RoundCornersTopLeft | ImDrawFlags_RoundCornersTopRight));

                    text_width = ImGui::CalcTextSize(this->class_name.c_str()).x;
                    text_pos_left_upper =
                        ImVec2(module_center.x - (text_width / 2.0f), module_rect_min.y + (style.ItemSpacing.y / 2.0f));
                    draw_list->AddText(text_pos_left_upper, COLOR_TEXT, this->class_name.c_str());

                    text_width = ImGui::CalcTextSize(this->name.c_str()).x;
                    text_pos_left_upper = module_center - ImVec2((text_width / 2.0f),
                                                              ((button_count > 0) ? (line_height * 0.6f) : (0.0f)));

                    draw_list->AddText(text_pos_left_upper, COLOR_TEXT, this->name.c_str());
                }

                if (button_count > 0) {
                    float item_x_offset = (ImGui::GetFrameHeight() / 2.0f);
                    float item_y_offset = (line_height / 2.0f);
                    if (!state.interact.module_show_label) {
                        item_y_offset = -(line_height / 2.0f);
                    }
                    if (button_count == 2) {
                        item_x_offset = ImGui::GetFrameHeight() + (0.5f * style.ItemSpacing.x * state.canvas.zooming);
                    } else if (button_count == 3) {
                        item_x_offset = (0.5 * ImGui::GetFrameHeight()) + ImGui::GetFrameHeight() +
                                        (style.ItemSpacing.x * state.canvas.zooming);
                    }
                    ImGui::SetCursorScreenPos(module_center + ImVec2(-item_x_offset, item_y_offset));

                    if (graph_entry_button) {
                        bool is_graph_entry = this->IsGraphEntry();
                        if (ImGui::RadioButton("###graph_entry_switch", is_graph_entry)) {
                            if (!is_graph_entry) {
                                state.interact.module_graphentry_changed = vislib::math::Ternary::TRI_TRUE;
                            } else {
                                state.interact.module_graphentry_changed = vislib::math::Ternary::TRI_FALSE;
                            }
                        }
                        ImGui::SetItemAllowOverlap();
                        if (this->gui_hovered) {
                            std::string tooltip_label;
                            if (is_graph_entry) {
                                tooltip_label = tooltip_label + "Graph Entry '" + this->graph_entry_name + "'";
                            } else {
                                tooltip_label = "No Graph Entry";
                            }
                            ImGui::PushFont(state.canvas.gui_font_ptr);
                            this->gui_other_item_hovered |= this->gui_tooltip.ToolTip(tooltip_label);
                            ImGui::PopFont();
                        }
                        ImGui::SameLine(0.0f, style.ItemSpacing.x * state.canvas.zooming);
                    }

                    // Param Button
                    if (parameter_button) {
                        ImVec2 param_popup_pos = ImGui::GetCursorScreenPos();
                        if (this->gui_selected) {
                            this->gui_param_child_show = ((state.interact.module_param_child_position.x > 0.0f) &&
                                                          (state.interact.module_param_child_position.y > 0.0f));
                        } else {
                            this->gui_param_child_show = false;
                        }
                        if (ImGui::ArrowButton("###parameter_toggle",
                                ((this->gui_param_child_show) ? (ImGuiDir_Down) : (ImGuiDir_Up))) &&
                            this->gui_hovered) {
                            this->gui_param_child_show = !this->gui_param_child_show;
                            if (this->gui_param_child_show) {
                                state.interact.module_param_child_position = param_popup_pos;
                                state.interact.module_param_child_position.x += ImGui::GetFrameHeight();
                            } else {
                                state.interact.module_param_child_position = ImVec2(-1.0f, -1.0f);
                            }
                        }
                        ImGui::SetItemAllowOverlap();
                        if (this->gui_hovered) {
                            ImGui::PushFont(state.canvas.gui_font_ptr);
                            this->gui_other_item_hovered |= this->gui_tooltip.ToolTip("Parameters");
                            ImGui::PopFont();
                        }
                    }
#ifdef MEGAMOL_USE_PROFILING
                    // Profiling Button
                    if (profiling_button) {
                        if (parameter_button || graph_entry_button) {
                            ImGui::SameLine(0.0f, style.ItemSpacing.x * state.canvas.zooming);
                        }

                        // Lazy loading of performance button texture
                        if (!this->gui_profiling_button.IsLoaded()) {
                            this->gui_profiling_button.LoadTextureFromFile(GUI_FILENAME_TEXTURE_PROFILING_BUTTON);
                        }
                        auto button_size = ImGui::GetFrameHeight();
                        this->profiling_button_position =
                            ImVec2(ImGui::GetCursorScreenPos().x + button_size / 2.0f, module_rect_max.y);
                        if (this->gui_profiling_button.Button("", ImVec2(button_size, button_size)) &&
                            this->gui_hovered) {
                            this->show_profiling_data = !this->show_profiling_data;
                            this->gui_update = true;
                            if (this->show_profiling_data) {
                                state.interact.profiling_show = true;
                            }
                        }
                        ImGui::SetItemAllowOverlap();
                        if (this->gui_hovered) {
                            ImGui::PushFont(state.canvas.gui_font_ptr);
                            this->gui_other_item_hovered |= this->gui_tooltip.ToolTip("Profiling");
                            ImGui::PopFont();
                        }
                        if (this->show_profiling_data) {
                            this->pause_profiling_history_update = state.interact.profiling_pause_update;
                        }
                    }
#endif // MEGAMOL_USE_PROFILING
                }

                // Draw Outline
                float border = ((!this->graph_entry_name.empty()) ? (4.0f) : (1.0f)) * megamol::gui::gui_scaling.Get() *
                               state.canvas.zooming;
                draw_list->AddRect(module_rect_min, module_rect_max, COLOR_MODULE_BORDER, GUI_RECT_CORNER_RADIUS,
                    ImDrawFlags_RoundCornersAll, border);
            }

            // CALL SLOTS ------------------------------------------------------
            for (auto& callslots_map : this->CallSlots()) {
                for (auto& callslot_ptr : callslots_map.second) {
                    callslot_ptr->Draw(phase, state);
                }
            }
            ImGui::PopID();
        }

    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


ImVec2 megamol::gui::Module::GetDefaultModulePosition(const GraphCanvas_t& canvas) {

    return ((ImVec2((2.0f * GUI_GRAPH_BORDER), (2.0f * GUI_GRAPH_BORDER)) + // ImGui::GetTextLineHeightWithSpacing()) +
                (canvas.position - canvas.offset)) /
            canvas.zooming);
}


void megamol::gui::Module::update(const GraphItemsState_t& state) {

    ImGuiStyle& style = ImGui::GetStyle();

    // WIDTH
    bool graph_entry_button = this->is_view;
    bool parameter_button = (!this->parameters.empty());
    bool profiling_button = false;
#ifdef MEGAMOL_USE_PROFILING
    profiling_button = true;
#endif // MEGAMOL_USE_PROFILING
    float button_count = (graph_entry_button) ? (1.0f) : (0.0f);
    button_count += (parameter_button) ? (1.0f) : (0.0f);
    button_count += (profiling_button) ? (1.0f) : (0.0f);
    float button_width = button_count * (ImGui::GetTextLineHeightWithSpacing() + style.ItemSpacing.x);

    float class_width = 0.0f;
    float max_label_length = 0.0f;
    if (state.interact.module_show_label) {
        class_width = ImGui::CalcTextSize(this->class_name.c_str()).x;
        float name_length = ImGui::CalcTextSize(this->name.c_str()).x;
        max_label_length = name_length;
    }

    max_label_length = std::max(max_label_length, button_width);
    max_label_length /= state.canvas.zooming;
    float max_slot_name_length = 0.0f;
    if (state.interact.callslot_show_label) {
        for (auto& callslots_map : this->CallSlots()) {
            for (auto& callslot_ptr : callslots_map.second) {
                max_slot_name_length =
                    std::max(ImGui::CalcTextSize(callslot_ptr->Name().c_str()).x, max_slot_name_length);
            }
        }
    }
    if (max_slot_name_length > 0.0f) {
        max_slot_name_length = (2.0f * max_slot_name_length / state.canvas.zooming) + (1.0f * GUI_SLOT_RADIUS);
    }
    float module_width = std::max((class_width / state.canvas.zooming), (max_label_length + max_slot_name_length)) +
                         (3.0f * GUI_SLOT_RADIUS);

    // HEIGHT
    float line_height = (ImGui::GetTextLineHeightWithSpacing() / state.canvas.zooming);
    auto max_slot_count =
        std::max(this->CallSlots(CallSlotType::CALLEE).size(), this->CallSlots(CallSlotType::CALLER).size());
    float module_slot_height =
        line_height + (static_cast<float>(max_slot_count) * (GUI_SLOT_RADIUS * 2.0f) * 1.5f) + GUI_SLOT_RADIUS;
    float text_button_height = (line_height * ((state.interact.module_show_label) ? (4.0f) : (1.0f)));
    float module_height = std::max(module_slot_height, text_button_height);

    // Clamp to minimum size
    this->gui_size = ImVec2(std::max(module_width, (2.0f * megamol::gui::gui_scaling.Get())),
        std::max(module_height, (1.0f * megamol::gui::gui_scaling.Get())));

    // UPDATE all Call Slots ---------------------
    for (auto& slot_pair : this->CallSlots()) {
        for (auto& slot : slot_pair.second) {
            slot->Update(state);
        }
    }
}


void megamol::gui::Module::SetName(const std::string& mod_name) {

    this->name = mod_name;
    for (auto& p : this->parameters) {
        p.SetParentModuleName(this->FullName());
    }
}


void megamol::gui::Module::SetGroupName(const std::string& gr_name) {

    this->group_name = gr_name;
    for (auto& p : this->parameters) {
        p.SetParentModuleName(this->FullName());
    }
}


bool megamol::gui::Module::ParametersVisible() {

    return this->gui_param_groups.ParametersVisible(this->parameters);
}


bool megamol::gui::Module::StateToJSON(nlohmann::json& inout_json) {

    // Parameter Groups
    bool retval = this->gui_param_groups.StateToJSON(inout_json, this->FullName());
    // Parameters
    for (auto& param : this->parameters) {
        retval &= param.StateToJSON(inout_json, param.FullName());
    }
    return retval;
}


bool megamol::gui::Module::StateFromJSON(const nlohmann::json& in_json) {

    // Parameter Groups
    bool retval = this->gui_param_groups.StateFromJSON(in_json, this->FullName());
    // Parameters
    for (auto& param : this->parameters) {
        retval &= param.StateFromJSON(in_json, param.FullName());
        param.ForceSetGUIStateDirty();
    }
    return retval;
}


#ifdef MEGAMOL_USE_PROFILING

void megamol::gui::Module::AppendPerformanceData(frontend_resources::PerformanceManager::frame_type frame,
    const frontend_resources::PerformanceManager::timer_entry& entry) {
    if (!this->pause_profiling_history_update) {
        switch (entry.api) {
        case frontend_resources::PerformanceManager::query_api::CPU:
            this->cpu_perf_history[entry.handle].push_sample(frame, entry.frame_index,
                std::chrono::duration<double, std::milli>(entry.duration.time_since_epoch()).count());
            break;
        case frontend_resources::PerformanceManager::query_api::OPENGL:
            this->gl_perf_history[entry.handle].push_sample(frame, entry.frame_index,
                std::chrono::duration<double, std::milli>(entry.duration.time_since_epoch()).count());
            break;
        }
    }
}


void megamol::gui::Module::DrawProfiling(GraphItemsState_t& state) {

    ImGui::BeginTabBar("profiling", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyScroll);
    ProfilingUtils::ProxyVector histories;
    histories.append(this->cpu_perf_history);
    histories.append(this->gl_perf_history);
    for (size_t i = 0; i < histories.size(); i++) {
        auto& tab_label = histories[i].get_name();
        if (ImGui::BeginTabItem(tab_label.c_str(), nullptr, ImGuiTabItemFlags_None)) {

            static ProfilingUtils::MetricType display_idx = ProfilingUtils::MetricType::MINMAXAVG;
            static ImPlotAxisFlags y_flags = ImPlotAxisFlags_AutoFit;
            ProfilingUtils::MetricDropDown(display_idx, y_flags);

            if (ImGui::BeginTable(("table_" + tab_label).c_str(), 2,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableColumnFlags_NoResize,
                    ImVec2(0.0f, 0.0f))) {
                ImGui::TableSetupColumn(("column_" + tab_label).c_str(), ImGuiTableColumnFlags_WidthStretch);

                ProfilingUtils::PrintTableRow(
                    "Min Time", histories[i].window_statistics(core::MultiPerformanceHistory::metric_type::MIN,
                                    core::MultiPerformanceHistory::metric_type::MIN));
                ProfilingUtils::PrintTableRow(
                    "Average Time", histories[i].window_statistics(core::MultiPerformanceHistory::metric_type::AVERAGE,
                                        core::MultiPerformanceHistory::metric_type::AVERAGE));
                ProfilingUtils::PrintTableRow(
                    "Max Time", histories[i].window_statistics(core::MultiPerformanceHistory::metric_type::MAX,
                                    core::MultiPerformanceHistory::metric_type::MAX));
                ProfilingUtils::PrintTableRow("Max Samples / Frame",
                    static_cast<int>(histories[i].window_statistics(core::MultiPerformanceHistory::metric_type::MAX,
                        core::MultiPerformanceHistory::metric_type::COUNT)));
                ProfilingUtils::PrintTableRow("Num Samples", static_cast<int>(histories[i].samples()));

                ImGui::EndTable();
            }

            ProfilingUtils::DrawPlot("History", ImVec2(ImGui::GetContentRegionAvail().x, MODULE_PROFILING_PLOT_HEIGHT),
                y_flags, display_idx, histories[i]);

            ImGui::EndTabItem();
            y_flags = 0;
        }
    }
    ImGui::EndTabBar();

    auto new_profiling_window_height = std::max(1.0f, ImGui::GetCursorPosY());
    if (this->profiling_window_height != new_profiling_window_height) {
        this->profiling_window_height = new_profiling_window_height;
        this->gui_update = true;
    }
}

#endif // MEGAMOL_USE_PROFILING
