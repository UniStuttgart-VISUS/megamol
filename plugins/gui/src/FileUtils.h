/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#if _HAS_CXX17
#    include <filesystem> // directory_iterator
namespace fsns = std::filesystem;
#else
// WINDOWS
#    ifdef _WIN32
#        include <filesystem>
#    else
// LINUX
#        include <experimental/filesystem>
#    endif
namespace fsns = std::experimental::filesystem;
#endif

#include <fstream>
#include <iostream>
#include <string>

#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"

#include "vislib/sys/AbstractReaderWriterLock.h"


namespace megamol {
namespace gui {

/** Type for filesystem paths. */
typedef fsns::path PathType;

/**
 * Check if given file or directory exists.
 *
 * @param path  The file or directory path.
 */
inline bool PathExists(PathType path) { return fsns::exists(path); }

/**
 * Check if file exists and has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
inline bool HasExistingFileExtension(PathType path, std::string ext) {
    if (!fsns::exists(static_cast<PathType>(path))) {
        return false;
    }
    return (path.extension().generic_string() == ext);
}

/**
 * Check if file has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
inline bool HasFileExtension(PathType path, std::string ext) { return (path.extension().generic_string() == ext); }

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
    for (const auto& entry : fsns::recursive_directory_iterator(searchPath)) {
        if (entry.path().filename().generic_string() == file) {
            foundPath = entry.path().generic_string();
            break;
        }
    }
    return foundPath;
}


/**
 * Save project to file.
 *
 * @param projectFilename ...
 * @param moduleLock      ...
 * @param rootModule      ...
 * @param coreInstance    ...
 *
 * @return ...
 */
inline bool SaveProjectFile(std::string projectFilename, vislib::sys::AbstractReaderWriterLock& moduleLock,
    std::shared_ptr<megamol::core::AbstractNamedObject> rootModule, megamol::core::CoreInstance* coreInstance) {

    if (coreInstance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[SaveProjectFile] CoreInstance is nullptr.");
        return false;
    }

    std::stringstream confInstances, confModules, confCalls, confParams;

    std::map<std::string, std::string> view_instances;
    // std::map<std::string, std::string> job_instances;
    {
        vislib::sys::AutoLock lock(moduleLock);
        megamol::core::AbstractNamedObjectContainer::ptr_type anoc =
            megamol::core::AbstractNamedObjectContainer::dynamic_pointer_cast(rootModule);
        // int job_counter = 0;
        for (auto ano = anoc->ChildList_Begin(); ano != anoc->ChildList_End(); ++ano) {
            auto vi = dynamic_cast<megamol::core::ViewInstance*>(ano->get());
            auto ji = dynamic_cast<megamol::core::JobInstance*>(ano->get());
            if (vi && vi->View()) {
                std::string vin = vi->Name().PeekBuffer();
                view_instances[vi->View()->FullName().PeekBuffer()] = vin;
                vislib::sys::Log::DefaultLog.WriteInfo("[SaveProjectFile] Found view instance \"%s\" with view \"%s\".",
                    view_instances[vi->View()->FullName().PeekBuffer()].c_str(), vi->View()->FullName().PeekBuffer());
            }
            // if (ji && ji->Job()) {
            //    std::string jin = ji->Name().PeekBuffer();
            //    // todo: find job module! WTF!
            //    job_instances[jin] = std::string("job") + std::to_string(job_counter);
            //    vislib::sys::Log::DefaultLog.WriteInfo("ScreenShooter: found job instance \"%s\" with job \"%s\".",
            //        jin.c_str(), job_instances[jin].c_str());
            //    ++job_counter;
            //}
        }

        const auto fun = [&confInstances, &confModules, &confCalls, &confParams, &view_instances](
                             megamol::core::Module* mod) {
            if (view_instances.find(mod->FullName().PeekBuffer()) != view_instances.end()) {
                confInstances << "mmCreateView(\"" << view_instances[mod->FullName().PeekBuffer()] << "\",\""
                              << mod->ClassName() << "\",\"" << mod->FullName().PeekBuffer() << "\")\n";
            } else {
                // todo: jobs??
                confModules << "mmCreateModule(\"" << mod->ClassName() << "\",\"" << mod->FullName().PeekBuffer()
                            << "\")\n";
            }
            megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator si =
                     mod->ChildList_Begin();
                 si != se; ++si) {
                const auto slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                if (slot) {
                    const auto bp = slot->Param<megamol::core::param::ButtonParam>();
                    if (!bp) {
                        std::string val = slot->Parameter()->ValueString().PeekBuffer();
                        //// caution: value strings could contain unescaped quotes, so fix that:
                        // std::string from = "\"";
                        // std::string to = "\\\"";
                        // size_t start_pos = 0;
                        // while ((start_pos = val.find(from, start_pos)) != std::string::npos) {
                        //    val.replace(start_pos, from.length(), to);
                        //    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
                        //}
                        confParams << "mmSetParamValue(\"" << slot->FullName() << "\",[=[" << val << "]=])\n";
                    }
                }
                const auto cslot = dynamic_cast<megamol::core::CallerSlot*>((*si).get());
                if (cslot) {
                    const megamol::core::Call* c =
                        const_cast<megamol::core::CallerSlot*>(cslot)->CallAs<megamol::core::Call>();
                    if (c != nullptr) {
                        confCalls << "mmCreateCall(\"" << c->ClassName() << "\",\""
                                  << c->PeekCallerSlot()->Parent()->FullName().PeekBuffer()
                                  << "::" << c->PeekCallerSlot()->Name().PeekBuffer() << "\",\""
                                  << c->PeekCalleeSlot()->Parent()->FullName().PeekBuffer()
                                  << "::" << c->PeekCalleeSlot()->Name().PeekBuffer() << "\")\n";
                    }
                }
            }
        };
        coreInstance->EnumModulesNoLock(nullptr, fun);
    }

    auto confstr = std::string("-- MegaMol Project File\n\n") + confInstances.str() + "\n" + confModules.str() + "\n" +
                   confCalls.str() + "\n" + confParams.str() + "\n";

    try {
        std::ofstream file;
        file.open(projectFilename);
        if (file.good()) {
            file << confstr.c_str();
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError("[SaveProjectFile] Couldn't create project file.");
            file.close();
            return false;
        }
    } catch (...) {
    }

    return true;
}


} // namespace gui
} // namespace megamol
