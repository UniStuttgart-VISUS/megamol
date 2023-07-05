/*
 * FileBrowserWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "FileBrowserWidget.h"
#include "gui_utils.h"
#include "imgui_stdlib.h"
#include "mmcore/utility/FileUtils.h"
#include "mmcore/utility/String.h"
#include "widgets/ButtonWidgets.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::core::param;


megamol::gui::FileBrowserWidget::FileBrowserWidget()
        : current_directory_str()
        , current_file_str()
        , path_changed(false)
        , valid_directory(false)
        , valid_file(false)
        , append_ending_str()
        , file_errors()
        , file_warnings()
        , child_directories()
        , save_gui_state(false)
        , save_all_param_values(false)
        , label_uid_map()
        , search_widget()
        , tooltip() {

    std::srand(std::time(nullptr));
}


bool megamol::gui::FileBrowserWidget::Button_Select(
    std::string& inout_filename, const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags) {

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

    return this->PopUp_Select("Select File", inout_filename, open_popup_select_file, extensions, flags);
}


bool megamol::gui::FileBrowserWidget::popup(FileBrowserWidget::DialogMode mode, const std::string& label,
    std::string& inout_filename, bool& inout_open_popup, const FilePathParam::Extensions_t& extensions,
    FilePathParam::Flags_t flags, bool& inout_save_gui_state, bool& inout_save_all_param_values) {

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
                if ((ei + 1) != extensions.end()) {
                    popup_label += " ";
                }
            }
            popup_label += ")";
        }
        popup_label += "###fbw" + this->label_uid_map[label];

        if (inout_open_popup) {
            this->validate_split_path(inout_filename, flags, this->current_directory_str, this->current_file_str);
            this->validate_directory(flags, this->current_directory_str);
            this->validate_file(mode, extensions, flags, this->current_directory_str, this->current_file_str);
            this->path_changed = true;

            ImGui::OpenPopup(popup_label.c_str());
            // Set initial window size of pop up
            ImGui::SetNextWindowSize(
                ImVec2((400.0f * megamol::gui::gui_scaling.Get()), (500.0f * megamol::gui::gui_scaling.Get())));
            inout_open_popup = false;

            this->search_widget.ClearSearchString();

            this->save_gui_state = inout_save_gui_state;
            this->save_all_param_values = inout_save_all_param_values;
        }

        bool open = true;
        if (ImGui::BeginPopupModal(popup_label.c_str(), &open, ImGuiWindowFlags_None)) {

            bool apply = false;

            // Path ---------------------------------------------------
            auto last_file_path_str = this->current_directory_str;
            if (ImGui::ArrowButton("arrow_home_dir", ImGuiDir_Right)) {
                this->current_directory_str = std::filesystem::current_path().generic_u8string();
            }
            this->tooltip.ToolTip("Working Directory", ImGui::GetID("arrow_home_dir"), 0.5f, 5.0f);
            ImGui::SameLine();
            if (ImGui::ArrowButton("arrow_up_dir", ImGuiDir_Up)) {
                this->current_directory_str = this->get_parent_path(this->current_directory_str);
            }
            this->tooltip.ToolTip("Up", ImGui::GetID("arrow_up_dir"), 0.5f, 5.0f);
            ImGui::SameLine();
            ImGui::InputText("Directory", &this->current_directory_str, ImGuiInputTextFlags_AutoSelectAll);
            this->tooltip.ToolTip(this->current_directory_str);
            if (last_file_path_str != this->current_directory_str) {
                this->validate_directory(flags, this->current_directory_str);
                this->path_changed = true;
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
            // Footer Height: 1x save gui state line + 2x line for button + 2x max log lines
            float footer_height = ImGui::GetFrameHeightWithSpacing() * ((mode != DIALOGMODE_SAVE) ? (2.0f) : (3.0f)) +
                                  (ImGui::GetTextLineHeightWithSpacing() * 2.0f);
            float child_select_height = (ImGui::GetContentRegionAvail().y - footer_height);
            ImGui::BeginChild(
                "files_list_child_window", ImVec2(0.0f, child_select_height), true, ImGuiWindowFlags_None);

            if (this->valid_directory) {
                // Parent directory selectable
                if (ImGui::Selectable("..", false, select_flags)) {
                    this->current_directory_str = this->get_parent_path(this->current_directory_str);
                    this->path_changed = true;
                }

                // Only update child paths when path changed.
                if (this->path_changed) {
                    // Reset scrolling
                    ImGui::SetScrollY(0.0f);
                    // Update child paths
                    this->child_directories.clear();
                    std::vector<ChildData_t> paths;
                    std::vector<ChildData_t> files;
                    try {
                        auto tmp_file_path = std::filesystem::u8path(this->current_directory_str);
                        for (const auto& entry : std::filesystem::directory_iterator(tmp_file_path)) {
                            if (status_known(status(entry.path()))) {
                                bool is_dir = std::filesystem::is_directory(entry.path());
                                if (is_dir) {
                                    paths.emplace_back(ChildData_t(entry.path(), is_dir));
                                } else {
                                    files.emplace_back(ChildData_t(entry.path(), is_dir));
                                }
                            }
                        }
                    } catch (...) {}

                    // Sort path and file names case insensitive alphabetically ascending
                    std::sort(paths.begin(), paths.end(), [&](ChildData_t const& a, ChildData_t const& b) {
                        std::string a_str = a.first.filename().generic_u8string();
                        core::utility::string::ToUpperAscii(a_str);
                        std::string b_str = b.first.filename().generic_u8string();
                        core::utility::string::ToUpperAscii(b_str);
                        return (a_str < b_str);
                    });
                    std::sort(files.begin(), files.end(), [&](ChildData_t const& a, ChildData_t const& b) {
                        std::string a_str = a.first.filename().generic_u8string();
                        core::utility::string::ToUpperAscii(a_str);
                        std::string b_str = b.first.filename().generic_u8string();
                        core::utility::string::ToUpperAscii(b_str);
                        return (a_str < b_str);
                    });

                    for (auto& path : paths) {
                        this->child_directories.emplace_back(path);
                    }
                    for (auto& file : files) {
                        this->child_directories.emplace_back(file);
                    }

                    this->path_changed = false;
                }

                // Files and directories ----------------
                for (const auto& path_pair : this->child_directories) {

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
                                select_label.c_str(), (select_label == this->current_file_str), select_flags)) {
                            last_file_path_str = this->current_directory_str;
                            auto new_path = path_pair.first.generic_u8string();
                            this->validate_split_path(
                                new_path, flags, this->current_directory_str, this->current_file_str);
                            this->validate_file(
                                mode, extensions, flags, this->current_directory_str, this->current_file_str);
                            if (last_file_path_str != this->current_directory_str) {
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
            if (!this->file_warnings.empty()) {
                ImGui::TextColored(GUI_COLOR_TEXT_WARN, this->file_warnings.c_str());
            }
            if (!this->file_errors.empty()) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, this->file_errors.c_str());
            }
            float max_log_lines = 2.0f;
            ImGui::SetCursorScreenPos(cursor_pos + ImVec2(0.0f, max_log_lines * ImGui::GetTextLineHeightWithSpacing()));

            // File name ------------------------
            if (!(flags & FilePathParam::Internal_NoExistenceCheck)) {
                ImGui::BeginDisabled();
            }
            auto last_file_name_str = this->current_file_str;

            std::string input_label = "File Name";
            if ((flags & FilePathParam::Flag_Any) == FilePathParam::Flag_Any) {
                input_label = "File or Directory Name";
            } else if (flags & FilePathParam::Flag_Directory) {
                input_label = "Directory Name";
            }
            if (ImGui::InputText(input_label.c_str(), &this->current_file_str,
                    ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll)) {
                apply = true;
            }
            if (!(flags & FilePathParam::Internal_NoExistenceCheck)) {
                ImGui::EndDisabled();
            }
            if (last_file_name_str != this->current_file_str) {
                this->validate_file(mode, extensions, flags, this->current_directory_str, this->current_file_str);
            }

            // Optional save GUI state option ------------
            if (mode == DIALOGMODE_SAVE) {
                megamol::gui::ButtonWidgets::ToggleButton("Save GUI state", this->save_gui_state);
                this->tooltip.Marker("Check this option to also save all settings affecting the GUI.");
                ImGui::SameLine();
                megamol::gui::ButtonWidgets::ToggleButton("Save all parameter values", this->save_all_param_values);
                this->tooltip.Marker("Check this option to save all paramter values, not only the changed.");
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

            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }

            // Return complete file path --------------------------------------
            if (apply && this->valid_directory && this->valid_file) {

                auto dir = std::filesystem::u8path(this->current_directory_str);
                auto file = std::filesystem::u8path(this->current_file_str);
                std::filesystem::path tmp_path;
                if ((flags & FilePathParam::Flag_Any) == FilePathParam::Flag_Any) {
                    tmp_path = dir / file;
                    if (dir.stem() == file.stem()) {
                        tmp_path = dir;
                    }
                } else if (flags & FilePathParam::Flag_File) {
                    // Assemble final file name
                    this->current_file_str += this->append_ending_str;
                    tmp_path = dir / std::filesystem::u8path(this->current_file_str);
                    /// TODO tmp_path = std::filesystem::relative(tmp_path, std::filesystem::current_path());
                } else if (flags & FilePathParam::Flag_Directory) {
                    tmp_path = dir / file;
                    if (dir.stem() == file.stem()) {
                        tmp_path = dir;
                    }
                }
                inout_filename = tmp_path.generic_u8string();
                inout_save_gui_state = this->save_gui_state;
                inout_save_all_param_values = this->save_all_param_values;
                ImGui::CloseCurrentPopup();
                retval = true;
            }

            ImGui::EndGroup();
            // --------------------------------------------------------------------

            ImGui::EndPopup();
        }

    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
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
    const std::string& in_path, FilePathParam::Flags_t flags, std::string& out_dir, std::string& out_file) const {

    try {
        out_dir.clear();
        out_file.clear();

        auto out_path = std::filesystem::u8path(in_path);

        if (out_path.empty()) {
            out_path = std::filesystem::current_path();
        }

        if ((flags & FilePathParam::Flag_Any) == FilePathParam::Flag_Any) {
            if ((status_known(status(out_path)) && is_directory(out_path))) {
                out_dir = out_path.generic_u8string();
                out_file = out_path.filename().generic_u8string();
            } else {
                out_dir = out_path.parent_path().generic_u8string();
                out_file = out_path.filename().generic_u8string();
            }
        } else if (flags & FilePathParam::Flag_File) {
            if ((status_known(status(out_path)) && is_directory(out_path))) {
                out_dir = out_path.generic_u8string();
                if (!(flags & FilePathParam::Internal_NoExistenceCheck)) {
                    out_file.clear();
                }
            } else {
                out_dir = out_path.parent_path().generic_u8string();
                out_file = out_path.filename().generic_u8string();
            }
        } else if (flags & FilePathParam::Flag_Directory) {
            if ((status_known(status(out_path)) && is_directory(out_path))) {
                out_dir = out_path.generic_u8string();
                out_file = out_path.filename().generic_u8string();
            } else {
                out_dir = out_path.parent_path().generic_u8string();
                out_file = out_path.filename().generic_u8string();
            }
        }
        if (out_dir.empty()) {
            out_dir = ".";
        }

    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        out_dir.clear();
        out_file.clear();
        return false;
    }
    return true;
}


void megamol::gui::FileBrowserWidget::validate_directory(FilePathParam::Flags_t flags, const std::string& path_str) {

    try {
        auto tmp_path = std::filesystem::u8path(path_str);
        this->valid_directory =
            (status_known(status(tmp_path)) && is_directory(tmp_path) && (tmp_path.root_name() != tmp_path));
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


void megamol::gui::FileBrowserWidget::validate_file(FileBrowserWidget::DialogMode mode,
    const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags, const std::string& directory_str,
    const std::string& file_str) {

    try {
        this->file_errors.clear();
        this->file_warnings.clear();
        this->valid_file = true;
        this->append_ending_str.clear();

        auto dir = std::filesystem::u8path(directory_str);
        auto file = std::filesystem::u8path(file_str);
        auto tmp_filepath = dir / file;
        if (flags & FilePathParam::Flag_Directory) {
            if (dir.stem() == file.stem()) {
                tmp_filepath = dir;
            }
        }
        auto error_flags = FilePathParam::ValidatePath(tmp_filepath, extensions, flags);
        if (error_flags != 0) {
            this->valid_file = false;
        }
        if (error_flags & FilePathParam::Flag_File) {
            this->file_errors += "Expecting file.\n";
        }
        if (error_flags & FilePathParam::Flag_Directory) {
            this->file_errors += "Expecting directory.\n";
        }
        if (error_flags & FilePathParam::Internal_NoExistenceCheck) {
            this->file_errors += "Path does not exist.\n";
        }
        if (error_flags & FilePathParam::Internal_RestrictExtension) {
            std::string log_exts;
            FilePathParam::Extensions_t tmp_exts;
            for (auto& ext : extensions) {
                tmp_exts.emplace_back("." + ext);
                core::utility::string::ToLowerAscii(tmp_exts.back());
                log_exts += "'" + tmp_exts.back() + "' ";
            }
            if ((mode == DIALOGMODE_SAVE) && tmp_filepath.extension().empty()) {
                // Automatically append first extension
                if (this->file_errors.empty()) {
                    if (!tmp_exts.empty()) {
                        this->file_warnings += "Appending required file extension '" + tmp_exts[0] + "'\n";
                        this->append_ending_str = tmp_exts[0];
                    }
                    this->valid_file = true;
                }
            } else {
                this->file_errors += "File does not have required extension: " + log_exts + "\n";
            }
        }

        if ((mode == DIALOGMODE_SAVE) && (flags & FilePathParam::Internal_NoExistenceCheck) &&
            this->file_errors.empty()) {
            if (flags & FilePathParam::Flag_File) {
                // Warn if file already exists
                tmp_filepath = dir / std::filesystem::u8path(file_str + this->append_ending_str);
                if (std::filesystem::exists(tmp_filepath) && std::filesystem::is_regular_file(tmp_filepath)) {
                    this->file_warnings += "Overwriting existing file.\n";
                }
            } else if (flags & FilePathParam::Flag_Directory) {
                // Warn if directory already exists
                if (std::filesystem::exists(tmp_filepath) && std::filesystem::is_directory(tmp_filepath)) {
                    this->file_warnings += "Overwriting existing directory.\n";
                }
            }
        }
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return;
    }
}


std::string FileBrowserWidget::get_parent_path(const std::string& dir) const {

    try {
        auto retdir = this->get_absolute_path(dir);
        auto parent_dir = std::filesystem::u8path(retdir);
        if (parent_dir.has_parent_path() && parent_dir.has_relative_path()) {
            retdir = parent_dir.parent_path().generic_u8string();
        }
        return retdir;
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return std::string();
    }
}


std::string megamol::gui::FileBrowserWidget::get_absolute_path(const std::string& dir) const {

    try {
        auto retval = std::filesystem::u8path(dir);
        if ((retval.generic_u8string() == "..") || (retval.generic_u8string() == ".")) {
            retval = absolute(retval);
#if (_MSC_VER < 1916) /// XXX Fixed/No more required since VS 2019
            if (retval.has_parent_path()) {
                retval = retval.parent_path();
                if ((retval.generic_u8string() == "..") && retval.has_parent_path()) {
                    retval = retval.parent_path();
                }
            }
#endif // _MSC_VER > 1916
        }
        return retval.generic_u8string();
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return std::string();
    }
}
