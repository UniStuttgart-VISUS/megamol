/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <string>

namespace megamol::core::utility::platform {

    class RuntimeInfo {
    public:

        static std::string GetOsInfo() {
            if (m_os_info.empty()) {
                get_os_info();
            }
            return m_os_info;
        }

        static std::string GetRuntimeLibraries() {
            if (m_runtime_libraries.empty()) {
                get_runtime_libraries();
            }
            return m_runtime_libraries;
        }


    private:
        static void get_runtime_libraries();
        static void get_os_info();
        static std::string execute(const std::string& cmd);

        inline static std::string m_runtime_libraries;
        inline static std::string m_os_info;
    };
}
