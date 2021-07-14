/*
 * FileBrowserWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "FileBrowserWidget.h"
#include "gui_utils.h"
#include "imgui_stdlib.h"
#include "widgets/ButtonWidgets.h"


using namespace megamol;
using namespace megamol::gui;


megamol::gui::FileBrowserWidget::FileBrowserWidget()
        : search_widget()
        , file_name_str()
        , file_path_str()
        , path_changed(false)
        , valid_directory(false)
        , valid_file(false)
        , valid_ending()
        , file_error()
        , file_warning()
        , child_paths()
        , save_gui_state(vislib::math::Ternary::TRI_UNKNOWN)
        , label_uid_map()
        , return_path(PATHMODE_ABSOLUTE)
        , tooltip() {

    std::srand(std::time(nullptr));
}


bool megamol::gui::FileBrowserWidget::Button_Select(const std::vector<std::string>& extensions,
    std::string& inout_filename, bool force_absolute, const std::string& project_path) {

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

    float width = half_edge_length * 0.7f;
    float height = width * 0.7f;

    ImVec2 center = widget_start_pos + ImVec2(half_edge_length, half_edge_length);
    ImVec2 upper_left = center - ImVec2(width, height);
    ImVec2 lower_right = center + ImVec2(width, height);
    ImVec4 color_front = style.Colors[ImGuiCol_ButtonActive];
    color_front.w = 1.0f;
    draw_list->AddRectFilled(upper_left, lower_right, ImGui::ColorConvertFloat4ToU32(color_front), 1.0f);

    center += ImVec2(half_edge_length * 0.25f, half_edge_length * 0.25f);
    upper_left = center - ImVec2(width, height);
    lower_right = center + ImVec2(width, height);
    color_front = style.Colors[ImGuiCol_ButtonHovered];
    color_front.w = 1.0f;
    draw_list->AddRectFilled(upper_left, lower_right, ImGui::ColorConvertFloat4ToU32(color_front), 1.0f);

    ImVec2 rect = ImVec2(edge_length, edge_length);
    bool open_popup_select_file = ImGui::InvisibleButton("special_button", rect);

    ImGui::EndChild();
    ImGui::PopStyleColor();

    return this->PopUp_Select(
        "Select File", extensions, open_popup_select_file, inout_filename, force_absolute, project_path);
}


std::string megamol::gui::FileBrowserWidget::get_absolute_path(const std::string& in_path_str) const {

    auto retval_str = in_path_str;
    if ((in_path_str == "..") || (in_path_str == ".")) {
        stdfs::path retval = static_cast<stdfs::path>(in_path_str);
        retval = stdfs::absolute(retval);
#if (_MSC_VER < 1916) /// XXX Fixed/No more required since VS 2019
        if (retval.has_parent_path()) {
            retval = retval.parent_path();
            if ((in_path_str == "..") && retval.has_parent_path()) {
                retval = retval.parent_path();
            }
        }
#endif // _MSC_VER > 1916
        retval_str = retval.generic_u8string();
        gui_utils::Utf8Decode(retval_str);
    }
    return retval_str;
}


bool megamol::gui::FileBrowserWidget::popup(DialogMode mode, const std::string& label,
    const std::vector<std::string>& extensions, bool& inout_open_popup, std::string& inout_filename,
    bool force_absolute_path, const std::string& project_path, vislib::math::Ternary& inout_save_gui_state) {

    bool retval = false;

    try {
        // Generate UID independent of label name
        if (this->label_uid_map.find(label) == this->label_uid_map.end()) {
            this->label_uid_map[label] = std::to_string(std::rand());
        }
        std::string popup_label = label;
        if (!extensions.empty()) {
            popup_label += " (";
            for (auto ei = extensions.begin(); ei != extensions.end(); ei++) {
                popup_label += "." + (*ei);
                if (ei + 1 != extensions.end()) {
                    popup_label += " ";
                }
            }
            popup_label += ")";
        }
        popup_label += "###fbw" + this->label_uid_map[label];

        if (inout_open_popup) {

            // Browse to given file name path
            this->validate_split_path(inout_filename, this->file_path_str, this->file_name_str);
            this->validate_directory(this->file_path_str);
            this->validate_file(this->file_name_str, extensions, mode);
            this->path_changed = true;

            ImGui::OpenPopup(popup_label.c_str());
            // Set initial window size of pop up
            ImGui::SetNextWindowSize(
                ImVec2((400.0f * megamol::gui::gui_scaling.Get()), (500.0f * megamol::gui::gui_scaling.Get())));
            inout_open_popup = false;

            this->search_widget.ClearSearchString();
            this->save_gui_state = inout_save_gui_state;
        }

        bool open = true;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, ImGuiWindowFlags_None)) {

            bool apply = false;
            bool opt_relabspath = false; /// XXX (!force_absolute_path && (mode == DIALOGMODE_SELECT));

            // Path ---------------------------------------------------
            auto last_file_path_str = this->file_path_str;
            if (ImGui::ArrowButton("###arrow_home", ImGuiDir_Right)) {
                this->file_path_str = stdfs::current_path().generic_u8string();
            }
            this->tooltip.ToolTip("Working Directory", ImGui::GetID("###arrow_home"), 0.5f, 5.0f);
            ImGui::SameLine();
            if (ImGui::ArrowButton("###arrow_up_directory", ImGuiDir_Up)) {
                stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str);
                if (tmp_file_path.has_parent_path() && tmp_file_path.has_relative_path()) {
                    // Assuming that parent is still valid directory
                    this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                }
            }
            ImGui::SameLine();
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            gui_utils::Utf8Encode(this->file_path_str);
            ImGui::InputText("Directory", &this->file_path_str, ImGuiInputTextFlags_AutoSelectAll);
            gui_utils::Utf8Decode(this->file_path_str);
            if (last_file_path_str != this->file_path_str) {
                this->path_changed = true;
                this->validate_directory(this->file_path_str);
            }
            // Error message when path is no valid directory
            if (!this->valid_directory) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, "Invalid Directory");
            }

            // Search -------------------------------
            std::string help_test = "Case insensitive substring search in\nlisted file and directory names.";
            this->search_widget.Widget("guiwindow_parameter_earch", help_test);
            auto currentSearchString = this->search_widget.GetSearchString();

            // File browser selectables ---------------------------------------
            auto select_flags = ImGuiSelectableFlags_DontClosePopups;
            float footer_height =
                ImGui::GetFrameHeightWithSpacing() * ((inout_save_gui_state.IsUnknown())
                                                             ? (2.0f)
                                                             : (3.0f)) + // 1x save gui state line + 2x line for button
                (ImGui::GetTextLineHeightWithSpacing() * 2.0f);          // 2x max log lines
            if (opt_relabspath) {
                footer_height += ImGui::GetTextLineHeightWithSpacing() * ((project_path.empty()) ? (4.5f) : (6.0f));
            }
            float child_select_height = (ImGui::GetContentRegionAvail().y - footer_height);
            ImGui::BeginChild(
                "files_list_child_window", ImVec2(0.0f, child_select_height), true, ImGuiWindowFlags_None);

            if (this->valid_directory) {
                // Parent directory selectable
                std::string tag_parent("..");
                if (ImGui::Selectable(tag_parent.c_str(), false, select_flags)) {
                    stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str);
                    if (tmp_file_path.has_parent_path() && tmp_file_path.has_relative_path()) {
                        // Assuming that parent is still valid directory
                        this->file_path_str = tmp_file_path.parent_path().generic_u8string();
                        this->path_changed = true;
                    }
                }

                // Only update child paths when path changed.
                if (this->path_changed) {
                    // Reset scrolling
                    ImGui::SetScrollY(0.0f);

                    this->file_path_str = this->get_absolute_path(this->file_path_str);

                    // Update child paths
                    this->child_paths.clear();

                    std::vector<ChildData_t> paths;
                    std::vector<ChildData_t> files;
                    try {
                        stdfs::path tmp_file_path = static_cast<stdfs::path>(this->file_path_str);
                        for (const auto& entry : stdfs::directory_iterator(tmp_file_path)) {
                            if (stdfs::status_known(stdfs::status(entry.path()))) {
                                bool is_directory = stdfs::is_directory(entry.path());
                                if (is_directory) {
                                    paths.emplace_back(ChildData_t(entry.path(), is_directory));
                                } else {
                                    files.emplace_back(ChildData_t(entry.path(), is_directory));
                                }
                            }
                        }
                    } catch (...) {}

                    // Sort path case insensitive alphabetically ascending
                    std::sort(paths.begin(), paths.end(), [&](ChildData_t const& a, ChildData_t const& b) {
                        std::string a_str = a.first.filename().generic_u8string();
                        megamol::gui::gui_utils::StringToUpperCase(a_str);
                        std::string b_str = b.first.filename().generic_u8string();
                        megamol::gui::gui_utils::StringToUpperCase(b_str);
                        return (a_str < b_str);
                    });
                    std::sort(files.begin(), files.end(), [&](ChildData_t const& a, ChildData_t const& b) {
                        std::string a_str = a.first.filename().generic_u8string();
                        megamol::gui::gui_utils::StringToUpperCase(a_str);
                        std::string b_str = b.first.filename().generic_u8string();
                        megamol::gui::gui_utils::StringToUpperCase(b_str);
                        return (a_str < b_str);
                    });

                    for (auto& path : paths) {
                        this->child_paths.emplace_back(path);
                    }
                    for (auto& file : files) {
                        this->child_paths.emplace_back(file);
                    }

                    this->path_changed = false;
                }

                // Files and directories ----------------
                for (const auto& path_pair : this->child_paths) {

                    auto select_label = path_pair.first.filename().generic_u8string();
                    bool showSearchedParameter = true;
                    if (!currentSearchString.empty()) {
                        showSearchedParameter =
                            gui_utils::FindCaseInsensitiveSubstring(select_label, currentSearchString);
                    }
                    if (showSearchedParameter) {
                        // Different color for directories
                        if (path_pair.second) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                        }

                        if (ImGui::Selectable(
                                select_label.c_str(), (select_label == this->file_name_str), select_flags)) {
                            last_file_path_str = this->file_path_str;
                            auto new_path = path_pair.first.generic_u8string();
                            gui_utils::Utf8Decode(new_path);
                            this->validate_split_path(new_path, this->file_path_str, this->file_name_str);
                            this->validate_file(this->file_name_str, extensions, mode);
                            if (last_file_path_str != this->file_path_str) {
                                this->path_changed = true;
                            }
                        }

                        if (path_pair.second) {
                            ImGui::PopStyleColor();
                        }
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsItemHovered()) {
                            apply = true;
                        }
                    }
                }
            }
            ImGui::EndChild();

            // Widget group -------------------------------------------------------
            ImGui::BeginGroup();

            auto cursor_pos = ImGui::GetCursorScreenPos();
            // Error and warn messages ----------
            if (!this->file_warning.empty()) {
                ImGui::TextColored(GUI_COLOR_TEXT_WARN, this->file_warning.c_str());
            }
            if (!this->file_error.empty()) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, this->file_error.c_str());
            }
            float max_log_lines = 2.0f;
            ImGui::SetCursorScreenPos(cursor_pos + ImVec2(0.0f, max_log_lines * ImGui::GetTextLineHeightWithSpacing()));

            // File name ------------------------
            if (mode == DIALOGMODE_LOAD) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            auto last_file_name_str = this->file_name_str;
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            gui_utils::Utf8Encode(this->file_name_str);
            if (ImGui::InputText("File Name", &this->file_name_str,
                    ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
                apply = true;
            }
            gui_utils::Utf8Decode(this->file_name_str);
            if (mode == DIALOGMODE_LOAD) {
                ImGui::PopItemFlag();
            }
            if (last_file_name_str != this->file_name_str) {
                this->validate_file(this->file_name_str, extensions, mode);
            }

            // Optional save GUI state option ------------
            if (!inout_save_gui_state.IsUnknown()) {
                bool check = this->save_gui_state.IsTrue();
                megamol::gui::ButtonWidgets::ToggleButton("Save GUI state", check);
                this->save_gui_state =
                    ((check) ? (vislib::math::Ternary::TRI_TRUE) : (vislib::math::Ternary::TRI_FALSE));
                this->tooltip.Marker("Check this option to also save all settings affecting the GUI.");
            }

            // Relative output path options
            if (opt_relabspath) {
                ImGui::Separator();
                ImGui::TextUnformatted("Return Path: ");
                ImGui::AlignTextToFramePadding();
                // ImGui::SameLine();
                if (ImGui::RadioButton("Absolute", (this->return_path == PATHMODE_ABSOLUTE))) {
                    this->return_path = PATHMODE_ABSOLUTE;
                }
                // ImGui::SameLine();
                if (ImGui::RadioButton(
                        "Relative to Working Directory", (this->return_path == PATHMODE_RELATIVE_WORKING))) {
                    this->return_path = PATHMODE_RELATIVE_WORKING;
                }
                // ImGui::SameLine();
                if (!project_path.empty()) {
                    if (ImGui::RadioButton(
                            "Relative to Project File", (this->return_path == PATHMODE_RELATIVE_PROJECT))) {
                        this->return_path = PATHMODE_RELATIVE_PROJECT;
                    }
                }
                ImGui::Separator();
            }

            // Buttons --------------------------
            std::string button_label;
            gui_utils::PushReadOnly((!(this->valid_directory && this->valid_file)));
            if (mode == DIALOGMODE_SAVE) {
                button_label = "Save";
            } else if (mode == DIALOGMODE_LOAD) {
                button_label = "Load";
            } else if (mode == DIALOGMODE_SELECT) {
                button_label = "Select";
            }
            if (ImGui::Button(button_label.c_str())) {
                apply = true;
            }
            gui_utils::PopReadOnly((!(this->valid_directory && this->valid_file)));

            ImGui::SameLine();

            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
                ImGui::CloseCurrentPopup();
            }

            // Return complete file path --------------------------------------
            if (apply && this->valid_directory && this->valid_file) {

                // Assemble final file name
                this->file_name_str += this->valid_ending;
                stdfs::path tmp_file_path =
                    static_cast<stdfs::path>(this->file_path_str) / static_cast<stdfs::path>(this->file_name_str);

                // Check for desired path format
                if (opt_relabspath) {
                    if (this->return_path == PATHMODE_RELATIVE_PROJECT) {
                        if (!project_path.empty()) {
                            auto relative_project_dir = stdfs::path(project_path);
                            tmp_file_path = stdfs::relative(tmp_file_path, relative_project_dir);
                        }
                    } else if (this->return_path == PATHMODE_RELATIVE_WORKING) {
                        tmp_file_path = stdfs::relative(
                            tmp_file_path, stdfs::current_path()); /// XXX requires non-experimental filesystem support
                    } else {
                        tmp_file_path = stdfs::absolute(tmp_file_path);
                    }
                    if (tmp_file_path.empty()) {
                        tmp_file_path = static_cast<stdfs::path>(this->file_path_str) /
                                        static_cast<stdfs::path>(this->file_name_str);
                    }
                }

                inout_filename = tmp_file_path.generic_u8string();
                gui_utils::Utf8Decode(inout_filename);
                inout_save_gui_state = this->save_gui_state;
                ImGui::CloseCurrentPopup();
                retval = true;
            }

            ImGui::EndGroup();
            // --------------------------------------------------------------------

            ImGui::EndPopup();
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

    return retval;
}


bool megamol::gui::FileBrowserWidget::validate_split_path(
    const std::string& in_path_file, std::string& out_path, std::string& out_file) const {

    // Splitting file path into path string and file string
    try {
        out_path.clear();
        out_file.clear();
        stdfs::path out_path_file(in_path_file.c_str());
        if (out_path_file.empty()) {
            out_path_file = stdfs::current_path();
            out_path = out_path_file.generic_u8string();
        } else if ((stdfs::status_known(stdfs::status(out_path_file)) && stdfs::is_directory(out_path_file))) {
            out_path = out_path_file.generic_u8string();
        } else if (stdfs::status_known(stdfs::status(out_path_file)) && stdfs::is_regular_file(out_path_file)) {
            out_path = out_path_file.parent_path().generic_u8string();
            out_file = out_path_file.filename().generic_u8string();
            if (out_path.empty()) {
                out_path = ".";
            }
        } else {
            out_path = out_path_file.parent_path().generic_u8string();
            out_file = out_path_file.filename().generic_u8string();
            if (out_path.empty()) {
                out_path = ".";
            }
        }
        gui_utils::Utf8Decode(out_path);
        gui_utils::Utf8Decode(out_file);

    } catch (stdfs::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        out_path.clear();
        out_file.clear();
        return false;
    }
    return true;
}


void megamol::gui::FileBrowserWidget::validate_directory(const std::string& path_str) {

    // Validating existing directory
    try {
        stdfs::path tmp_path = static_cast<stdfs::path>(path_str);
        this->valid_directory = (stdfs::status_known(stdfs::status(tmp_path)) && stdfs::is_directory(tmp_path) &&
                                 (tmp_path.root_name() != tmp_path));
    } catch (stdfs::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::FileBrowserWidget::validate_file(const std::string& file_str,
    const std::vector<std::string>& extensions, megamol::gui::FileBrowserWidget::DialogMode mode) {

    // Validating given path
    try {
        std::string file_lower = file_str;
        megamol::gui::gui_utils::StringToLowerCase(file_lower);

        std::vector<std::string> exts_lower;
        for (auto& ext : extensions) {
            exts_lower.emplace_back("." + ext);
            megamol::gui::gui_utils::StringToLowerCase(exts_lower.back());
        }

        this->file_error.clear();
        this->file_warning.clear();
        this->valid_file = true;
        this->valid_ending.clear();

        stdfs::path tmp_file_path =
            static_cast<stdfs::path>(this->file_path_str) / static_cast<stdfs::path>(file_lower);

        if (mode == DIALOGMODE_SAVE) {

            // Warn when no file name is given
            if (file_lower.empty()) {
                this->file_warning += "Enter file name.\n";
                this->valid_file = false;
            } else {
                // Warn when file has not required extension
                if (!exts_lower.empty()) {
                    if (!megamol::core::utility::FileUtils::FileHasExtension<std::string>(file_lower, exts_lower[0])) {
                        this->file_warning += "Appending required file extension '" + exts_lower[0] + "'\n";
                        this->valid_ending = exts_lower[0];
                    }
                }
                std::string actual_filename = file_lower;
                actual_filename += this->valid_ending;
                tmp_file_path =
                    static_cast<stdfs::path>(this->file_path_str) / static_cast<stdfs::path>(actual_filename);

                // Warn when file already exists
                if (stdfs::exists(tmp_file_path) && stdfs::is_regular_file(tmp_file_path)) {
                    this->file_warning += "Overwriting existing file.\n";
                }
            }

            // Error when file is directory
            if (stdfs::is_directory(tmp_file_path)) {
                this->file_error += "Input is directory.\n";
                this->valid_file = false;
            }
        } else if (mode == DIALOGMODE_LOAD) {

            // Error when file has not required extension
            if (!exts_lower.empty()) {
                bool extension_found = false;
                for (auto& ext : exts_lower) {
                    if (megamol::core::utility::FileUtils::FileHasExtension<std::string>(file_lower, ext)) {
                        extension_found = true;
                    }
                }
                if (!extension_found) {
                    this->file_error += "Require file with extension: ";
                    for (auto ei = exts_lower.begin(); ei != exts_lower.end(); ei++) {
                        this->file_error += "'" + (*ei) + "'";
                        if (ei + 1 != exts_lower.end()) {
                            this->file_error += " ";
                        }
                    }
                    this->valid_ending.clear();
                    this->valid_file = false;
                }
            }

            // Error when file is directory
            if (stdfs::is_directory(tmp_file_path)) {
                this->file_error += "Selection is directory.\n";
                this->valid_file = false;
            }
        } else if (mode == DIALOGMODE_SELECT) {

            // Warning when file is directory
            if (stdfs::is_directory(tmp_file_path)) {
                this->file_warning += "Selection is directory.\n";
            }
        }

    } catch (stdfs::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}
