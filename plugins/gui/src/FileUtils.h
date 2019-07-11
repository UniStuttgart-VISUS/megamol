/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/// Exeption for the cluster "Stampede" running CentOS.
/// C++ filesystem support is not working:
/// /opt/apps/gcc/7.3.0/bin/ld:
/// /opt/apps/gcc/7.3.0/lib/gcc/x86_64-pc-linux-gnu/7.3.0/../../../../lib64/libstdc++fs.a(ops.o)(.text._ZN9__gnu_cxx13stdio_filebufIcSt11char_traitsIcEED2Ev[_ZN9__gnu_cxx13stdio_filebufIcSt11char_traitsIcEED5Ev]+0x3):
/// unresolvable R_X86_64_NONE relocation against symbol `_ZTVSt13basic_filebufIcSt11char_traitsIcEE@@GLIBCXX_3.4'
/// /opt/apps/gcc/7.3.0/bin/ld: final link failed: Nonrepresentable section on output
#ifdef GUI_USE_FILEUTILS

#    ifndef MEGAMOL_GUI_FILEUTILS_H_INCLUDED
#        define MEGAMOL_GUI_FILEUTILS_H_INCLUDED
#        if (defined(_MSC_VER) && (_MSC_VER > 1000))
#            pragma once
#        endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#        if defined(_WIN32) && defined(_MANAGED)
#            pragma managed(push, off)
#        endif /* defined(_WIN32) && defined(_MANAGED) */

#        include <string>

#        if _HAS_CXX17
#            include <filesystem> // directory_iterator
namespace ns_fs = std::filesystem;
#        else
// WINDOWS
#            ifdef _WIN32
#                include <filesystem>
#            else
// LINUX
#                include <experimental/filesystem>
#            endif
namespace ns_fs = std::experimental::filesystem;
#        endif


namespace megamol {
namespace gui {

/** Type for filesystem paths. */
typedef ns_fs::path PathType;

/**
 * Check if given file or directory exists.
 *
 * @param path  The file or directory path.
 */
inline bool PathExists(PathType path) { return ns_fs::exists(path); }

/**
 * Check if filename exists and has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
inline bool HasFileExtension(PathType path, std::string ext) {
    if (!ns_fs::exists(static_cast<PathType>(path))) {
        return false;
    }
    return (path.extension().generic_string() == ext);
}

/**
 * Search recursively for file or path beginning at given directory.
 *
 * @param file          The file to search for.
 * @param searchPath    The path of a directory as start for recursive search.
 *
 * @return              The complete path of the found file, empty string otherwise.
 */
inline std::string SearchFileRecursive(std::string file, PathType searchPath) {
    std::string foundPath;
    for (const auto& entry : ns_fs::recursive_directory_iterator(searchPath)) {
        if (entry.path().filename().generic_string() == file) {
            foundPath = entry.path().generic_string();
            break;
        }
    }
    return foundPath;
}

} // namespace gui
} // namespace megamol

#    endif // MEGAMOL_GUI_FILEUTILS_H_INCLUDED

#endif // GUI_USE_FILEUTILS