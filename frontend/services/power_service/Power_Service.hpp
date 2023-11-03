/*
 * Power_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <fstream>
#include <list>
#include <sstream>
#include <thread>

#include "mmcore/MegaMolGraph.h"

#include "AbstractFrontendService.hpp"

#include "RuntimeInfo.h"

#include "power/CryptToken.h"
#include "power/MetaData.h"
#include "power/ParallelPortTrigger.h"
#include "power/RTXInstruments.h"
#include "power/SamplerCollection.h"
#include "power/SignalBroker.h"
#include "power/StringContainer.h"
#include "power/Trigger.h"
#include "power/Utility.h"

#include "PowerCallbacks.h"

#include <power_overwhelming/hmc8015_sensor.h>
#include <power_overwhelming/rtx_instrument.h>

#include <sol/sol.hpp>

namespace megamol {
namespace frontend {

inline std::string gen_hmc_filename(unsigned int const cnt) {
    return std::string("PWR_") + std::to_string(cnt);
}

inline std::string gen_hmc_filename(std::string const& fix) {
    return fix + std::string(".CSV");
}

class Power_Service final : public AbstractFrontendService {
public:
    struct Config {
        std::string lpt = "lpt1";
        bool write_to_files = false;
        std::string folder = "./";
        power::StringContainer* str_container;
    };

    std::string serviceName() const override {
        return "Power_Service";
    }

    Power_Service();

    ~Power_Service();

    // init service with input config data, e.g. init GLFW with OpenGL and open window with certain decorations/hints
    // if init() fails return false (this will terminate program execution), on success return true
    bool init(void* configPtr) override;

    void close() override;

    std::vector<FrontendResource>& getProvidedResources() override;

    const std::vector<std::string> getRequestedResourceNames() const override;

    void setRequestedResources(std::vector<FrontendResource> resources) override;

    void updateProvidedResources() override;

    void digestChangedRequestedResources() override;

    void resetProvidedResources() override;

    void preGraphRender() override;

    void postGraphRender() override;

private:
    // this can hold references to the resources (i.e. structs) we provide to others, e.g. you may fill this and return it in getProvidedResources()
    // provided resources will be queried by the system only once,
    // there is no requirement to store the resources in a vector the whole time, you just need to return such a vector in getProvidedResources()
    // but you need to store the actual resource objects you provide and manage
    // note that FrontendResource wraps a void* to the objects you provide, thus your resource objects will not be copied, but they will be referenced
    // (however the FrontendResource objects themselves will be copied)
    std::vector<FrontendResource> m_providedResourceReferences;

    // names of resources you request for your service can go here
    // requested resource names will be queried by the system only once,
    // there is no requirement to store the names in a vector the whole time, you just need to return such a vector in getRequestedResourceNames()
    std::vector<std::string> m_requestedResourcesNames;

    // you may store the resources you requested in this vector by filling it when your setRequestedResources() gets called
    // the resources provided to you by the system match the names you requested in getRequestedResourceNames() and are expected to reference actual existing objects
    // the sorting of resources matches the order of your requested resources names, you can use this to directly index into the vector provided by setRequestedResources()
    // if every service follows the rules the provided resources should be valid existing objects, thus you can use them directly without error or nullptr checking,
    // but we in the end we must blindly rely on the std::any in FrontendResource to hold the struct or type you expect it to hold
    // (or else std::any will throw a bad type cast exception that should terminate program execution.
    // you do NOT catch or check for that exception or need to care for it in any way!)
    std::vector<FrontendResource> m_requestedResourceReferences;

    frontend_resources::PowerCallbacks callbacks_;

    std::vector<visus::power_overwhelming::rtx_instrument> rtx_instr_;

    std::unique_ptr<power::SamplersCollectionWrapper> samplers;

    std::unordered_map<std::string, visus::power_overwhelming::hmc8015_sensor> hmc_sensors_;

    void fill_lua_callbacks();

    void reset_segment_range(std::chrono::milliseconds const& range);

    void reset_measurement();

    void write_sample_buffers(std::size_t seg_cnt);

    //    void sb_pre_trg() {
    //        //static auto freq = static_cast<double>(megamol::power::get_highres_timer_freq());
    //        do_buffer_ = true;
    //#ifdef WIN32
    //        FILETIME f;
    //        GetSystemTimePreciseAsFileTime(&f);
    //        ULARGE_INTEGER tv;
    //        tv.HighPart = f.dwHighDateTime;
    //        tv.LowPart = f.dwLowDateTime;
    //        //sb_qpc_offset_100ns_ = static_cast<double>(megamol::power::get_highres_timer()) / freq * 10000000.0 - tv.QuadPart;
    //        sb_qpc_offset_100ns_ = tv.QuadPart;
    //#else
    //#endif
    //    }

    void seg_post_trg() {
        ++seg_cnt_;
    }

    void sb_pre_trg() {
        do_buffer_ = true;

        samplers->visit(&power::ISamplerCollection::StartRecording);

        /*if (nvml_samplers_)
            nvml_samplers_->StartRecording();
        if (adl_samplers_)
            adl_samplers_->StartRecording();
        if (emi_samplers_)
            emi_samplers_->StartRecording();
        if (msr_samplers_)
            msr_samplers_->StartRecording();
        if (tinker_samplers_)
            tinker_samplers_->StartRecording();*/
    }

    /*void sb_sgn_trg(std::tuple<std::chrono::system_clock::time_point, int64_t> const& ts) {
        sb_qpc_offset_100ns_ = std::get<1>(ts);
    }*/

    void sb_post_trg() {
        do_buffer_ = false;

        samplers->visit(&power::ISamplerCollection::StopRecording);

        /*if (nvml_samplers_)
            nvml_samplers_->StopRecording();
        if (adl_samplers_)
            adl_samplers_->StopRecording();
        if (emi_samplers_)
            emi_samplers_->StopRecording();
        if (msr_samplers_)
            msr_samplers_->StopRecording();
        if (tinker_samplers_)
            tinker_samplers_->StopRecording();*/

        /*auto t = std::thread(std::bind(&Power_Service::write_sample_buffers, this, seg_cnt_));
        t.detach();*/

        if (write_to_files_) {
            write_sample_buffers(seg_cnt_);
        }
    }

    void hmc_pre_trg() {
        for (auto& [n, s] : hmc_sensors_) {
            s.log_file(gen_hmc_filename(seg_cnt_).c_str(), true, false);
            auto const path_size = s.log_file((char*)nullptr, 0);
            std::string path;
            path.resize(path_size);
            s.log_file(path.data(), path.size());
            if (!path.empty()) {
                path.resize(path.size() - 1);
            }
            std::regex reg(R"(^\"(\w+)\", INT$)");
            std::smatch match;
            if (std::regex_match(path, match, reg)) {
                hmc_paths_[n].push_back(gen_hmc_filename(match[1]));
            }
        }
        for (auto& [n, s] : hmc_sensors_) {
            s.log(true);
        }
    }

    void hmc_post_trg() {
        for (auto& [n, s] : hmc_sensors_) {
            s.log(false);
        }
    }

    std::tuple<std::string, std::string, power::value_map_t> parse_hmc_file(std::string hmc_file) {
        // some lines have a leading '\r'
        hmc_file.erase(
            std::remove_if(std::begin(hmc_file), std::end(hmc_file), [](auto const& c) { return c == '\r'; }));

        std::regex count_reg(R"(#Actual Count;(\d+)\s*)");
        std::regex date_reg(R"(#Date;([\d|-]+)\s*)");
        std::smatch match;

        std::stringstream hmc_stream(hmc_file);
        std::stringstream meta_stream;
        std::stringstream csv_stream;

        power::value_map_t vals;

        std::string line;

        int num_of_rows = 0;
        int line_count = 0;

        std::string date_str;

        std::vector<std::string> col_names;

        while (std::getline(hmc_stream, line)) {
            if (line[0] == '#') {
                // meta information
                meta_stream << line << '\n';
                if (std::regex_match(line, match, count_reg)) {
                    num_of_rows = std::stoi(match[1].str());
                }
                if (std::regex_match(line, match, date_reg)) {
                    date_str = match[1];
                }
            } else {
                if (line[0] != '\n') {
                    // csv data
                    if (num_of_rows == 0)
                        break;
                    if (line_count > num_of_rows)
                        break;
                    if (line_count == 0) {
                        // title line
                        std::string val_str;
                        auto sstream = std::istringstream(line);
                        while (std::getline(sstream, val_str, ';')) {
                            col_names.push_back(val_str);
                            if (val_str.find("Timestamp") != std::string::npos) {
                                vals[val_str] = power::timeline_t{};
                                std::get<power::timeline_t>(vals[val_str]).reserve(num_of_rows);
                            } else {
                                vals[val_str] = power::samples_t{};
                                std::get<power::samples_t>(vals[val_str]).reserve(num_of_rows);
                            }
                        }
                    } else {
                        // data line
                        std::string val_str;
                        std::vector<std::string> val_strs;
                        val_strs.reserve(col_names.size());
                        auto sstream = std::istringstream(line);
                        while (std::getline(sstream, val_str, ';')) {
                            val_strs.push_back(val_str);
                        }

                        if (val_strs.size() != col_names.size()) {
                            throw std::runtime_error("unexpected number of values in line");
                        }

                        for (std::size_t i = 0; i < val_strs.size(); ++i) {
                            if (col_names[i].find("Timestamp") != std::string::npos) {
                                // parse UTC timestamp with fractional seconds
                                auto const ms_pos = val_str.find('.');
                                int64_t t_ms = 0;
                                std::string time_str;
                                if (ms_pos == std::string::npos) {
                                    // timestamp without ms part
                                    time_str = val_str;
                                } else {
                                    time_str = std::string(val_str.begin(), val_str.begin() + ms_pos);
                                    auto const ms_str = std::string(val_str.begin() + ms_pos + 1, val_str.end());
                                    t_ms = std::stoi(ms_str);
                                }
                                std::chrono::utc_clock::time_point tp;
                                std::istringstream time_stream(date_str + "T" + time_str);
                                if (std::chrono::from_stream(time_stream, "%FT%T", tp)) {
                                    auto const ts = (power::convert_systemtp2ft(std::chrono::utc_clock::to_sys(tp)) +
                                                     std::chrono::duration_cast<power::filetime_dur_t>(
                                                         std::chrono::milliseconds(t_ms)))
                                                        .count();
                                    std::get<power::timeline_t>(vals.at(col_names[i])).push_back(ts);
                                } else {
                                    throw std::runtime_error("could not parse UTC time");
                                }
                            } else {
                                // data
                                if (!val_strs[i].empty())
                                    std::get<power::samples_t>(vals.at(col_names[i])).push_back(std::stof(val_strs[i]));
                                else
                                    std::get<power::samples_t>(vals.at(col_names[i]))
                                        .push_back(std::numeric_limits<float>::signaling_NaN());
                            }
                        }
                    }
                    csv_stream << line << '\n';
                    ++line_count;
                }
            }
        }

        return std::make_tuple(meta_stream.str(), csv_stream.str(), vals);
    }

    void hmc_fin_trg() {
        if (write_to_files_) {
            std::regex reg(R"(#Actual Count;(\d+)\s*)");
            std::smatch match;
            for (auto& [n, s] : hmc_sensors_) {
                if (hmc_paths_[n].size() != seg_cnt_)
                    continue;
                for (unsigned int hmc_m = 0; hmc_m < seg_cnt_; ++hmc_m) {
                    //auto blob = s.copy_file_from_instrument(gen_hmc_filename(hmc_m).c_str(), false);
                    auto blob = s.copy_file_from_instrument(hmc_paths_[n][hmc_m].c_str(), false);

                    auto hmc_file = std::string(blob.begin(), blob.end());

#if 0
                    hmc_file.erase(std::remove_if(
                        std::begin(hmc_file), std::end(hmc_file), [](auto const& c) { return c == '\r'; }));
                    std::stringstream hmc_stream(hmc_file);
                    std::stringstream meta_stream;
                    std::stringstream csv_stream;
                    std::string line;
                    int counter = 0;
                    int num_counts = 0;
                    while (std::getline(hmc_stream, line)) {
                        if (line[0] == '#') {
                            meta_stream << line << '\n';
                            if (std::regex_match(line, match, reg)) {
                                num_counts = std::stoi(match[1].str());
                            }
                        } else {
                            if (line[0] != '\n') {
                                if (counter > num_counts)
                                    break;
                                ++counter;
                                csv_stream << line << '\n';
                            }
                        }
                    }
#endif

                    auto const [meta_str, csv_str, vals] = parse_hmc_file(hmc_file);

                    auto const csv_path =
                        std::filesystem::path(write_folder_) / (n + "_s" + std::to_string(hmc_m) + ".csv");
                    /*auto const csv_path =
                        std::filesystem::path(write_folder_) /
                        (std::string("hmc") + std::to_string(counter) + "_s" + std::to_string(hmc_m) + ".csv");*/
                    std::ofstream file(csv_path.string(), std::ios::binary);
                    //file.write(blob.as<char const>(), blob.size());
                    file.write(csv_str.data(), csv_str.size());
                    file.close();
                    auto const meta_path =
                        std::filesystem::path(write_folder_) / (n + "_s" + std::to_string(hmc_m) + ".meta.csv");
                    file.open(meta_path.string(), std::ios::binary);
                    file.write(meta_str.data(), meta_str.size());
                    file.close();

                    auto const parquet_path =
                        std::filesystem::path(write_folder_) / (n + "_s" + std::to_string(hmc_m) + ".parquet");

                    power::ParquetWriter(parquet_path, vals, &meta_);
                }
            }
        }
        for (auto& [n, s] : hmc_sensors_) {
            if (hmc_paths_[n].size() != seg_cnt_) {
                hmc_paths_[n].clear();
                continue;
            }
            for (unsigned int hmc_m = 0; hmc_m < seg_cnt_; ++hmc_m) {
                //s.delete_file_from_instrument(gen_hmc_filename(hmc_m).c_str(), false);
                s.delete_file_from_instrument(hmc_paths_[n][hmc_m].c_str(), false);
            }
            hmc_paths_[n].clear();
        }
    }

    //std::unordered_map<std::string, std::string> exp_map_;

    //std::vector<float> examine_expression(std::string const& name, std::string const& exp_path, int s_idx);

    bool write_to_files_ = false;

    std::string write_folder_ = "./";

    std::unique_ptr<megamol::power::RTXInstruments> rtx_;

    bool do_buffer_ = false;

    std::size_t seg_cnt_ = 0;

    std::shared_ptr<megamol::power::Trigger> main_trigger_;

    //int64_t sb_qpc_offset_100ns_;

    power::StringContainer* str_cont_;

    frontend_resources::RuntimeInfo const* ri_;

    power::MetaData meta_;

    std::unique_ptr<power::CryptToken> dataverse_key_ = nullptr;

    struct dataverse_config_s {
        std::string base_path;
        std::string doi;
    } dataverse_config_;

    power::SignalBroker sbroker_;

    core::MegaMolGraph* megamolgraph_ptr_;

    std::unordered_map<std::string, std::vector<std::string>> hmc_paths_;
};

} // namespace frontend
} // namespace megamol

#endif
