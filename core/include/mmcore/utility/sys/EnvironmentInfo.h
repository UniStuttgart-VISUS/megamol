#pragma once
#include <string>

namespace megamol::core::utility::sys {

    class EnvironmentInfo {
    public:
        static std::string GetModuleInfo() {
            if (m_module_info.empty()) {
                init_module_info();
            }
            return m_module_info;
        }



    private:
        static std::string get_file_version(const char* path);
        static void init_module_info();

        inline static std::string m_module_info;
        inline static std::string m_os_info;
    };
}
