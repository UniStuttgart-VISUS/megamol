/*
 * FileBrowserWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_FILEBROWSERPOPUP_INCLUDED
#define MEGAMOL_GUI_FILEBROWSERPOPUP_INCLUDED


#include "FileUtils.h"
#include "GUIUtils.h"
#include "StringSearchWidget.h"


namespace megamol {
namespace gui {


/**
 * String search widget.
 */
class FileBrowserWidget {
public:
    FileBrowserWidget(void);

    ~FileBrowserWidget(void) = default;

    enum FileBrowserFlag { SAVE, LOAD, SELECT };

    /**
     * Draw file browser pop-up.
     *
     * @param flag                Flag inidicating intention of file browser dialog.
     * @param label               File browser label.
     * @param open_popup          Flag once(!) indicates opening of pop-up.
     * @param inout_filename      The file name of the file.
     *
     * @return True on success, false otherwise.
     */
    bool PopUp(FileBrowserFlag flag, const std::string& label_id, bool open_popup, std::string& inout_filename);

    /**
     * ImGui file browser button opening this file browser pop-up.
     *
     * @param inout_filename    The file name of the file.
     *
     * @return True on success, false otherwise.
     */
    bool Button(std::string& inout_filename);

private:
#ifdef GUI_USE_FILESYSTEM

    typedef std::pair<stdfs::path, bool> ChildData_t;

    // VARIABLES --------------------------------------------------------------

    StringSearchWidget search_widget;
    std::string file_name_str;
    std::string file_path_str;
    bool path_changed;
    bool valid_directory;
    bool valid_file;
    bool valid_ending;
    std::string file_error;
    std::string file_warning;
    // Keeps child path and flag whether child is director or not
    std::vector<ChildData_t> child_paths;
    size_t additional_lines;

    // FUNCTIONS --------------------------------------------------------------

    bool splitPath(const stdfs::path& in_file_path, std::string& out_path, std::string& out_file);
    void validateDirectory(const std::string& path_str);
    void validateFile(const std::string& file_str, FileBrowserFlag flag);

#endif // GUI_USE_FILESYSTEM
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_FileBrowserPopUp_INCLUDED
