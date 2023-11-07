#include "Utility.h"

#ifdef MEGAMOL_USE_POWER

#include <regex>

namespace megamol::power {

std::tuple<std::string, std::string, power::value_map_t> parse_hmc_file(std::string hmc_file) {
    // some lines have a leading '\r'
    hmc_file.erase(std::remove_if(std::begin(hmc_file), std::end(hmc_file), [](auto const& c) { return c == '\r'; }));

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
                                                     std::chrono::milliseconds(t_ms)) +
                                                 ft_offset)
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

} // namespace megamol::power

#endif
