/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <mutex>
#include <string>

#include "WMIUtil.h"

namespace megamol::core::utility::platform {

class RuntimeInfo {
public:
    static std::string GetHardwareInfo() {
        if (m_hardware_info.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_hardware_info();
        }
        return m_hardware_info;
    }

    static std::string GetOsInfo() {
        if (m_os_info.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_os_info();
        }
        return m_os_info;
    }

    static std::string GetRuntimeLibraries() {
        if (m_runtime_libraries.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_runtime_libraries();
        }
        return m_runtime_libraries;
    }

    static std::string GetSMBIOSInfo() {
        if (smbios_.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_smbios_info();
        }
        return smbios_;
    }

    static std::string GetCPUInfo() {
        if (cpu_.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_cpu_info();
        }
        return cpu_;
    }

    static std::string GetGPUInfo() {
        if (gpu_.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_gpu_info();
        }
        return gpu_;
    }

    static std::string GetOSInfo() {
        if (os_.empty()) {
            std::lock_guard<std::mutex> lock(write_mtx_);
            get_OS_info();
        }
        return os_;
    }


private:
    static void get_hardware_info();
    static void get_runtime_libraries();
    static void get_os_info();
    static std::string execute(const std::string& cmd);

    inline static std::string m_runtime_libraries;
    inline static std::string m_os_info;
    inline static std::string m_hardware_info;

    static void get_smbios_info(bool serial = false);
    static void get_cpu_info();
    static void get_gpu_info();
    static void get_OS_info();

    inline static std::string smbios_;
    inline static std::string cpu_;
    inline static std::string gpu_;
    inline static std::string os_;

    inline static std::mutex write_mtx_;

#ifdef _WIN32
    inline static WMIUtil wmi;
#endif
};
} // namespace megamol::core::utility::platform
