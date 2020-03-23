/*
 * PopUp.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIUtils.h"

#include "vislib/UTF8Encoder.h"

#include <imgui_stdlib.h>
#include <vector>


using namespace megamol;
using namespace megamol::gui;


GUIUtils::GUIUtils(void)
    : tooltip_time(0.0f)
    , tooltip_id(GUI_INVALID_ID)
    , search_focus(false)
    , search_string()
    , rename_string()
    , splitter_last_width(0.0f)
#ifdef GUI_USE_FILESYSTEM
    , file_name_str()
    , file_path_str()
    , child_paths()
    , path_changed(false)
    , valid_directory(false)
    , valid_file(false)
    , valid_ending(false)
    , file_error()
    , file_warning()
    , additional_lines(0)
#endif // GUI_USE_FILESYSTEM
{
}


void GUIUtils::HoverToolTip(const std::string& text, ImGuiID id, float time_start, float time_end) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();

    if (ImGui::IsItemHovered()) {
        bool show_tooltip = false;
        if (time_start > 0.0f) {
            if (this->tooltip_id != id) {
                this->tooltip_time = 0.0f;
                this->tooltip_id = id;
            } else {
                if ((this->tooltip_time > time_start) && (this->tooltip_time < (time_start + time_end))) {
                    show_tooltip = true;
                }
                this->tooltip_time += io.DeltaTime;
            }
        } else {
            show_tooltip = true;
        }

        if (show_tooltip) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::Text(text.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    } else {
        if ((time_start > 0.0f) && (this->tooltip_id == id)) {
            this->tooltip_time = 0.0f;
        }
    }
}


void GUIUtils::HelpMarkerToolTip(const std::string& text, std::string label) {
    assert(ImGui::GetCurrentContext() != nullptr);

    if (!text.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled(label.c_str());
        this->HoverToolTip(text);
    }
}


bool megamol::gui::GUIUtils::MinimalPopUp(const std::string& caption, bool open_popup, const std::string& info_text,
    const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted) {

    bool retval = false;

    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::PushID(caption.c_str());

    if (open_popup) {
        ImGui::OpenPopup(caption.c_str());
        float max_width = std::max(this->TextWidgetWidth(caption), this->TextWidgetWidth(info_text));
        max_width += (style.ItemSpacing.x * 2.0f);
        ImGui::SetNextWindowSize(ImVec2(max_width, 0.0f));
    }
    if (ImGui::BeginPopupModal(caption.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
        retval = true;

        if (!info_text.empty()) {
            ImGui::Text(info_text.c_str());
        }

        if (!confirm_btn_text.empty()) {
            if (ImGui::Button(confirm_btn_text.c_str())) {
                confirmed = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
        }
        if (!abort_btn_text.empty()) {
            if (ImGui::Button(abort_btn_text.c_str())) {
                aborted = true;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}


bool megamol::gui::GUIUtils::RenamePopUp(const std::string& caption, bool open_popup, std::string& rename) {

    bool retval = false;

    ImGui::PushID(caption.c_str());

    if (open_popup) {
        this->rename_string = rename;
        ImGui::OpenPopup(caption.c_str());
    }
    if (ImGui::BeginPopupModal(caption.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string text_label = "New Name";
        auto flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;
        if (ImGui::InputText(text_label.c_str(), &this->rename_string, flags)) {
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        // Set focus on input text once (applied next frame)
        if (open_popup) {
            ImGuiID id = ImGui::GetID(text_label.c_str());
            ImGui::ActivateItem(id);
        }

        if (ImGui::Button("OK")) {
            rename = this->rename_string;
            retval = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::PopID();

    return retval;
}


float GUIUtils::TextWidgetWidth(const std::string& text) const {
    assert(ImGui::GetCurrentContext() != nullptr);

    ImVec2 pos = ImGui::GetCursorPos();
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
    ImGui::Text(text.c_str());
    ImGui::PopStyleVar();
    ImGui::SetCursorPos(pos);

    return ImGui::GetItemRectSize().x;
}


void megamol::gui::GUIUtils::ReadOnlyWigetStyle(bool set) {

    if (set) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    } else {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }
}


bool GUIUtils::Utf8Decode(std::string& str) const {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Decode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}


bool GUIUtils::Utf8Encode(std::string& str) const {
    vislib::StringA dec_tmp;
    if (vislib::UTF8Encoder::Encode(dec_tmp, vislib::StringA(str.c_str()))) {
        str = std::string(dec_tmp.PeekBuffer());
        return true;
    }
    return false;
}


bool megamol::gui::GUIUtils::StringSearch(const std::string& id, const std::string& help) {

    bool retval = false;

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::BeginGroup();
    ImGui::PushID(id.c_str());

    if (ImGui::Button("Clear")) {
        this->search_string = "";
    }
    ImGui::SameLine();

    // Set keyboard focus when hotkey is pressed
    if (this->search_focus) {
        ImGui::SetKeyboardFocusHere();
        this->search_focus = false;
    }

    std::string complete_label = "Search (?)";
    auto width = ImGui::GetContentRegionAvailWidth() - ImGui::GetCursorPosX() + 4.0f * style.ItemInnerSpacing.x -
                 this->TextWidgetWidth(complete_label);
    const float min_width = 50.0f;
    width = (width < min_width) ? (min_width) : width;
    ImGui::PushItemWidth(width);

    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    this->Utf8Encode(this->search_string);
    ImGui::InputText("Search", &this->search_string, ImGuiInputTextFlags_AutoSelectAll);
    this->Utf8Decode(this->search_string);
    if (ImGui::IsItemActive()) {
        retval = true;
    }

    ImGui::PopItemWidth();
    ImGui::SameLine();

    this->HelpMarkerToolTip(help);

    ImGui::PopID();
    ImGui::EndGroup();

    return retval;
}


bool megamol::gui::GUIUtils::VerticalSplitter(FixedSplitterSide fixed_side, float& size_left, float& size_right) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    const float thickness = 12.0f;

    bool split_vertically = true;
    float min_size = 1.0f; // >=1.0!
    float splitter_long_axis_size = ImGui::GetContentRegionAvail().y;

    float width_avail = ImGui::GetWindowSize().x - (2.0f * thickness);

    size_left = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? size_left : (width_avail - size_right));
    size_right = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (width_avail - size_left) : size_right);

    size_left = std::max(size_left, min_size);
    size_right = std::max(size_right, min_size);

    ImGuiWindow* window = ImGui::GetCurrentContext()->CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    if (fixed_side == GUIUtils::FixedSplitterSide::LEFT) {
        bb.Min =
            window->DC.CursorPos + (split_vertically ? ImVec2(size_left + 1.0f, 0.0f) : ImVec2(0.0f, size_left + 1.0f));
    } else if (fixed_side == GUIUtils::FixedSplitterSide::RIGHT) {
        bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2((width_avail - size_right) + 1.0f, 0.0f)
                                                          : ImVec2(0.0f, (width_avail - size_right) + 1.0f));
    }
    bb.Max = bb.Min + ImGui::CalcItemSize(split_vertically ? ImVec2(thickness - 4.0f, splitter_long_axis_size)
                                                           : ImVec2(splitter_long_axis_size, thickness - 4.0f),
                          0.0f, 0.0f);

    // ImGui::PushStyleColor(ImGuiCol_Separator, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]));
    // ImGui::PushStyleColor(
    //    ImGuiCol_SeparatorHovered, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonHovered]));
    // ImGui::PushStyleColor(ImGuiCol_SeparatorActive, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Button]));

    bool retval = ImGui::SplitterBehavior(
        bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, &size_left, &size_right, min_size, min_size, 0.0f, 0.0f);

    // ImGui::PopStyleColor(3);

    /// XXX Left mouse button (= 0) is not recognized poperly!? ...
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
        float consider_width = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? size_left : size_right);
        if (consider_width <= min_size) {
            size_left = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (this->splitter_last_width)
                                                                           : (width_avail - this->splitter_last_width));
            size_right = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (width_avail - this->splitter_last_width)
                                                                            : (this->splitter_last_width));
        } else {
            size_left = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (min_size) : (width_avail - min_size));
            size_right = ((fixed_side == GUIUtils::FixedSplitterSide::LEFT) ? (width_avail - min_size) : (min_size));
            this->splitter_last_width = consider_width;
        }
    }

    return retval;
}


bool megamol::gui::GUIUtils::PointCircleButton(const std::string& label) {

    bool retval = false;

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    float edge_length = ImGui::GetFrameHeight();
    float half_edge_length = edge_length / 2.0f;
    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    if (!label.empty()) {
        float text_x_offset_pos = edge_length + style.ItemInnerSpacing.x;
        ImGui::SetCursorScreenPos(widget_start_pos + ImVec2(text_x_offset_pos, 0.0f));
        ImGui::AlignTextToFramePadding();
        ImGui::Text(label.c_str());
        ImGui::SetCursorScreenPos(widget_start_pos);
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    ImGui::BeginChild("special_button_background", ImVec2(edge_length, edge_length), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    float thickness = edge_length / 5.0f;
    ImVec2 center = widget_start_pos + ImVec2(half_edge_length, half_edge_length);
    ImU32 color_front = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
    draw_list->AddCircleFilled(center, thickness, color_front, 12);
    draw_list->AddCircle(center, 2.0f * thickness, color_front, 12, (thickness / 2.0f));

    ImVec2 rect = ImVec2(edge_length, edge_length);
    retval = ImGui::InvisibleButton("special_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    return retval;
}


bool megamol::gui::GUIUtils::FileBrowserButton(std::string& inout_filename) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    float edge_length = ImGui::GetFrameHeight();
    float half_edge_length = edge_length / 2.0f;
    ImVec2 widget_start_pos = ImGui::GetCursorScreenPos();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_FrameBg]));
    ImGui::BeginChild("filebrowser_button_background", ImVec2(edge_length, edge_length), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    assert(draw_list != nullptr);

    float width = half_edge_length *0.7f;
    float height = width *0.7f;

    ImVec2 center = widget_start_pos + ImVec2(half_edge_length, half_edge_length);
    ImVec2 upper_left = center - ImVec2(width, height);
    ImVec2 lower_right = center + ImVec2(width, height);
    ImVec4 color_front = style.Colors[ImGuiCol_ButtonActive];
    color_front.w = 1.0f;
    draw_list->AddRectFilled(upper_left, lower_right, ImGui::ColorConvertFloat4ToU32(color_front), 1.0f);

    center += ImVec2(half_edge_length*0.25f, half_edge_length*0.25f);
    upper_left = center - ImVec2(width, height);
    lower_right = center + ImVec2(width, height);
    color_front = style.Colors[ImGuiCol_ButtonHovered];
    color_front.w = 1.0f;
    draw_list->AddRectFilled(upper_left, lower_right, ImGui::ColorConvertFloat4ToU32(color_front), 1.0f);

    ImVec2 rect = ImVec2(edge_length, edge_length);
    bool popup_select_file = ImGui::InvisibleButton("special_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    return this->FileBrowserPopUp(FileBrowserFlag::SELECT, "Select File", popup_select_file, inout_filename);
}


bool megamol::gui::GUIUtils::FileBrowserPopUp(
    megamol::gui::GUIUtils::FileBrowserFlag flag, const std::string& label, bool open_popup, std::string& inout_filename) {

    bool retval = false;

#ifdef GUI_USE_FILESYSTEM
    try {
        std::string popup_name = label;
        ImGui::PushID(label.c_str());

        if (open_popup) {
            // Check given file name path

            fsns::path tmp_file_path = static_cast<fsns::path>(inout_filename);
            if (tmp_file_path.empty() || !fsns::exists(tmp_file_path)) {
                tmp_file_path = fsns::current_path();
            }
            this->splitPath(tmp_file_path, this->file_path_str, this->file_name_str);
            this->validateDirectory(this->file_path_str);
            this->validateFile(this->file_name_str, flag);
            this->path_changed = true;

            ImGui::OpenPopup(popup_name.c_str());
            // Set initial window size of pop up
            ImGui::SetNextWindowSize(ImVec2(600.0f, 300.0f));
        }

        bool open = true;
        if (ImGui::BeginPopupModal(popup_name.c_str(), &open, ImGuiWindowFlags_None)) {

            bool apply = false;
            const auto error_color = ImVec4(0.9f, 0.0f, 0.0f, 1.0f);
            const auto warning_color = ImVec4(0.75f, 0.75f, 0.f, 1.0f);

            ImGuiStyle& style = ImGui::GetStyle();

            // Path ---------------------------------------------------
            auto last_file_path_str = this->file_path_str;
            if (ImGui::ArrowButton("###arrow_up_directory", ImGuiDir_Up)) {
                fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str);
                if (tmp_file_path.has_parent_path()) {
                    // Assuming that parent is still valid directory
                    this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                }
            }
            ImGui::SameLine();
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            this->Utf8Encode(this->file_path_str);
            ImGui::InputText("Directory", &this->file_path_str, ImGuiInputTextFlags_AutoSelectAll);
            this->Utf8Decode(this->file_path_str);
            if (last_file_path_str != this->file_path_str) {
                this->path_changed = true;
                this->validateDirectory(this->file_path_str);
            }
            // Error message when path is no valid directory
            if (!this->valid_directory) {
                ImGui::TextColored(error_color, "Invalid Directory");
            }

            // File browser selectables ---------------------------------------
            auto select_flags = ImGuiSelectableFlags_DontClosePopups;
            float child_select_height =
                (ImGui::GetContentRegionAvail().y - (ImGui::GetTextLineHeightWithSpacing() * this->additional_lines) -
                    ImGui::GetItemsLineHeightWithSpacing() * 2.0f);

            // Files and directories ----------------
            ImGui::BeginChild(
                "files_list_child_window", ImVec2(0.0f, child_select_height), true, ImGuiWindowFlags_None);

            if (this->valid_directory) {

                // Parent directory selectable
                std::string tag_parent = "..";
                if (ImGui::Selectable(tag_parent.c_str(), false, select_flags)) {
                    fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str);
                    if (tmp_file_path.has_parent_path()) {
                        // Assuming that parent is still valid directory
                        this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                        this->path_changed = true;
                    }
                }

                // Only update child paths when path changed.
                if (this->path_changed) {
                    this->child_paths.clear();
                    try {
                        fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str);
                        for (const auto& entry : fsns::directory_iterator(tmp_file_path)) {
                            if (fsns::status_known(fsns::status(entry.path()))) {
                                this->child_paths.emplace_back(
                                    ChildDataType(entry.path(), fsns::is_directory(entry.path())));
                            }
                        }
                    } catch (...) {
                    }

                    // Sort path case insensitive alphabetically ascending
                    std::sort(this->child_paths.begin(), this->child_paths.end(),
                        [](ChildDataType const& a, ChildDataType const& b) {
                            std::string a_str = a.first.filename().generic_u8string();
                            for (auto& c : a_str) c = std::toupper(c);
                            std::string b_str = b.first.filename().generic_u8string();
                            for (auto& c : b_str) c = std::toupper(c);
                            return (a_str < b_str);
                        });

                    this->path_changed = false;
                }

                // Files and directories of current directory
                for (const auto& path_pair : this->child_paths) {
                    // Different color for directories
                    if (path_pair.second) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                    }
                    auto label = path_pair.first.filename().generic_u8string();
                    if (ImGui::Selectable(label.c_str(), (label == this->file_name_str), select_flags)) {
                        last_file_path_str = this->file_path_str;
                        this->splitPath(path_pair.first, this->file_path_str, this->file_name_str);
                        this->validateFile(this->file_name_str, flag);
                        if (last_file_path_str != this->file_path_str) {
                            this->path_changed = true;
                        }
                    }
                    if (path_pair.second) {
                        ImGui::PopStyleColor();
                    }
                    if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                        apply = true;
                    }
                }
            }
            ImGui::EndChild();

            // Widget group -------------------------------------------------------
            ImGui::BeginGroup();

            // File name ------------------------
            if (flag == GUIUtils::FileBrowserFlag::LOAD) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            auto last_file_name_str = this->file_name_str;
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            this->Utf8Encode(this->file_name_str);
            if (ImGui::InputText("File Name", &this->file_name_str,
                    ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
                apply = true;
            }
            this->Utf8Decode(this->file_name_str);
            if (flag == GUIUtils::FileBrowserFlag::LOAD) {
                ImGui::PopItemFlag();
            }
            if (last_file_name_str != this->file_name_str) {
                this->validateFile(this->file_name_str, flag);
            }
            if (!this->file_warning.empty()) {
                ImGui::TextColored(warning_color, this->file_warning.c_str());
            }
            if (!this->file_error.empty()) {
                ImGui::TextColored(error_color, this->file_error.c_str());
            }

            // Buttons ------------------------------
            std::string label;
            if (flag == GUIUtils::FileBrowserFlag::SAVE) {
                label = "Save";
            } else if (flag == GUIUtils::FileBrowserFlag::LOAD) {
                label = "Load";
            } else if (flag == GUIUtils::FileBrowserFlag::SELECT) {
                label = "Select";
            }            
            // const auto err_btn_color = ImVec4(0.6f, 0.0f, 0.0f, 1.0f);
            // const auto er_btn_hov_color = ImVec4(0.9f, 0.0f, 0.0f, 1.0f);
            // ImGui::PushStyleColor(ImGuiCol_Button,
            //     !(this->valid_directory && this->valid_file) ? err_btn_color : style.Colors[ImGuiCol_Button]);
            // ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
            //     !(this->valid_directory && this->valid_file) ? er_btn_hov_color : style.Colors[ImGuiCol_ButtonHovered]);
            //ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);
            if (ImGui::Button(label.c_str())) {
                apply = true;
            }
            //ImGui::PopStyleColor(3);

            ImGui::SameLine();

            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            // Return complete file path --------------------------------------
            if (apply && this->valid_directory && this->valid_file) {
                // Appending required extension
                if (!this->valid_ending) {
                    this->file_name_str.append(".lua");
                }
                fsns::path tmp_file_path =
                    static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(this->file_name_str);
                inout_filename = tmp_file_path.generic_u8string();
                ImGui::CloseCurrentPopup();
                retval = true;
            }

            ImGui::EndGroup();
            // --------------------------------------------------------------------

            ImGui::EndPopup();
        }

        ImGui::PopID();

    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

#else // GUI_USE_FILESYSTEM
    vislib::sys::Log::DefaultLog.WriteError("Filesystem functionality not available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
#endif // GUI_USE_FILESYSTEM

    return retval;
}

#ifdef GUI_USE_FILESYSTEM

bool megamol::gui::GUIUtils::splitPath(const fsns::path& in_file_path, std::string& out_path, std::string& out_file) {

    // Splitting path into path string and file string
    try {
        if (fsns::is_regular_file(in_file_path)) {
            if (in_file_path.has_parent_path()) {
                out_path = in_file_path.parent_path().generic_u8string();
            }
            if (in_file_path.has_filename()) {
                out_file = in_file_path.filename().generic_u8string();
            }
        } else {
            out_path = in_file_path.generic_u8string();
            out_file.clear();
        }
    } catch (fsns::filesystem_error e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


void megamol::gui::GUIUtils::validateDirectory(const std::string& path_str) {

    // Validating directory
    try {
        fsns::path tmp_path = static_cast<fsns::path>(path_str);
        this->valid_directory = (fsns::status_known(fsns::status(tmp_path)) && fsns::is_directory(tmp_path));
    } catch (fsns::filesystem_error e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::GUIUtils::validateFile(const std::string& file_str, GUIUtils::FileBrowserFlag flag) {

    // Validating file
    try {
        const std::string ext = ".lua";
        this->file_error.clear();
        this->file_warning.clear();
        this->additional_lines = 0;
        this->valid_file = true;
        this->valid_ending = true;

        fsns::path tmp_file_path = static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(file_str);   
        
        if (flag == GUIUtils::FileBrowserFlag::SAVE) {
            // Warn when no file name is given
            if (file_str.empty()) {
                this->file_warning += "Enter file name.\n";
                this->additional_lines++;
                this->valid_file = false;
            } else {
                // Warn when file has not required extension
                if (!file::FileExtension<std::string>(file_str, ext)) {
                    this->file_warning += "Appending required file extension '" + ext + "'\n";
                    this->additional_lines++;
                    this->valid_ending = false;
                }
                std::string actual_filename = file_str;
                if (!this->valid_ending) {
                    actual_filename.append(ext);
                }
                tmp_file_path =
                    static_cast<fsns::path>(this->file_path_str) / static_cast<fsns::path>(actual_filename);                    
        
                // Warn when file already exists
                if (fsns::exists(tmp_file_path) && fsns::is_regular_file(tmp_file_path)) {
                    this->file_warning += "Overwriting existing file.\n";
                    this->additional_lines++;
                }               
            }
        } else if (flag == GUIUtils::FileBrowserFlag::LOAD) {
            // Error when file has not required extension
            if (!file::FileExtension<std::string>(file_str, ext)) {
                this->file_error += "File with extension '" + ext + "' required.\n";
                this->additional_lines++;
                this->valid_ending = false;
                this->valid_file = false;
            }           
        } else if (flag == GUIUtils::FileBrowserFlag::SELECT) {
            // nothing to check ...
        }

        // Error when file is directory
        if (fsns::is_directory(tmp_file_path)) {
            this->file_error += "File is directory.\n";
            this->additional_lines++;
            this->valid_file = false;
        }  

    } catch (fsns::filesystem_error e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}

#endif // GUI_USE_FILESYSTEM