/*
 * FileBrowserWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "HoverToolTip.h"
#include "StringSearchWidget.h"
#include "mmcore/param/FilePathParam.h"


using namespace megamol::core::param;


namespace megamol::gui {


/** ************************************************************************
 * File browser widget
 */
class FileBrowserWidget {
public:
    FileBrowserWidget();
    ~FileBrowserWidget() = default;

    /**
     * Show file browser pop-up.
     *
     * @param label                 The pop-up label.
     * @param inout_filename        The user selected file name. Provide file name to use as initial value.
     * @param inout_open_popup      Indicates whether to open the pop-up or not.
     * @param extensions            The required file extensions. Leave emtpy to allow all file extensions.
     * @param flags                 The flags defining the required file path.
     * @param inout_save_gui_state  The flag indicating whether to save the gui state or not.
     *
     * @return True on success, false otherwise.
     */
    bool PopUp_Save(const std::string& label, std::string& inout_filename, bool& inout_open_popup,
        const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags, bool& inout_save_gui_state,
        bool& inout_save_all_param_values) {
        return this->popup(DIALOGMODE_SAVE, label, inout_filename, inout_open_popup, extensions, flags,
            inout_save_gui_state, inout_save_all_param_values);
    }
    bool PopUp_Load(const std::string& label, std::string& inout_filename, bool& inout_open_popup,
        const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags) {
        bool dummy = false;
        return this->popup(DIALOGMODE_LOAD, label, inout_filename, inout_open_popup, extensions, flags, dummy, dummy);
    }
    bool PopUp_Select(const std::string& label, std::string& inout_filename, bool& inout_open_popup,
        const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags) {
        bool dummy = false;
        return this->popup(DIALOGMODE_SELECT, label, inout_filename, inout_open_popup, extensions, flags, dummy, dummy);
    }

    /**
     * ImGui file browser button opening this file browser pop-up.
     *
     * @param extensions       The filtered file extensions. Leave emtpy to allow all file extensions.
     * @param inout_filename   The user selected file name.
     * @param force_absolute   The flag that indicates to whether to return the absolute file path or to enable
     * relative path selection.
     * @param project_path     The path to the current project file that is used for determine relative path.
     *
     * @return True on success, false otherwise.
     */
    bool Button_Select(
        std::string& inout_filename, const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags);

private:
    typedef std::pair<std::filesystem::path, bool> ChildData_t;

    enum DialogMode { DIALOGMODE_SAVE, DIALOGMODE_LOAD, DIALOGMODE_SELECT };

    // VARIABLES --------------------------------------------------------------

    std::string current_directory_str;
    std::string current_file_str;

    bool path_changed;
    bool valid_directory;
    bool valid_file;
    std::string append_ending_str;

    std::string file_errors;
    std::string file_warnings;

    // Keeps child paths and flag whether child is directors or not
    std::vector<ChildData_t> child_directories;
    bool save_gui_state;
    bool save_all_param_values;
    std::map<std::string, std::string> label_uid_map;

    StringSearchWidget search_widget;
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    bool popup(DialogMode mode, const std::string& label, std::string& inout_filename, bool& inout_open_popup,
        const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags, bool& inout_save_gui_state,
        bool& inout_save_all_param_values);

    bool validate_split_path(
        const std::string& in_path, FilePathParam::Flags_t flags, std::string& out_dir, std::string& out_file) const;
    void validate_directory(FilePathParam::Flags_t flags, const std::string& directory_str);
    void validate_file(DialogMode mode, const FilePathParam::Extensions_t& extensions, FilePathParam::Flags_t flags,
        const std::string& directory_str, const std::string& file_str);

    std::string get_parent_path(const std::string& dir) const;
    std::string get_absolute_path(const std::string& dir) const;
};


} // namespace megamol::gui
