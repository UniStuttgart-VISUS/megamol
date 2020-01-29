#include "stdafx.h"
#include "glyph_data_reader.h"

#include "glyph_data_call.h"

#include "mmcore/Call.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/DataHash.h"

#include "vislib/sys/Log.h"

#include <fstream>
#include <iostream>
#include <string>

namespace megamol
{
    namespace flowvis
    {
        glyph_data_reader::glyph_data_reader() :
            output_slot("output_slot", "Output slot for glyphs"),
            file_path_slot("file_path_slot", "File path to the stored glyphs")
        {
            // Set connections and parameters
            this->output_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &glyph_data_reader::get_data);
            this->output_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &glyph_data_reader::get_extent);
            this->MakeSlotAvailable(&this->output_slot);

            this->file_path_slot << new core::param::FilePathParam("");
            this->MakeSlotAvailable(&this->file_path_slot);
        }

        glyph_data_reader::~glyph_data_reader()
        {
            this->Release();
        }

        bool glyph_data_reader::create()
        {
            return true;
        }

        void glyph_data_reader::release()
        { }

        bool glyph_data_reader::get_data(core::Call& call)
        {
            return true;
        }

        bool glyph_data_reader::get_extent(core::Call& call)
        {
            // Get call
            auto* glyph_call = dynamic_cast<glyph_data_call*>(&call);

            if (glyph_call != nullptr)
            {
                if (!this->file_path_slot.Param<core::param::FilePathParam>()->Value().IsEmpty() &&
                    this->file_path_slot.IsDirty())
                {
                    this->file_path_slot.ResetDirty();

                    // Open file
                    std::ifstream glyph_ifs(this->file_path_slot.Param<core::param::FilePathParam>()->Value(), std::ios_base::in | std::ios_base::binary);

                    if (glyph_ifs.good())
                    {
                        this->stored_data.points.clear();
                        this->stored_data.lines.clear();

                        this->stored_data.hash = 0;

                        std::size_t id = 0;

                        glyph_ifs.seekg(0, glyph_ifs.end);
                        const auto file_length = glyph_ifs.tellg();
                        glyph_ifs.seekg(0, glyph_ifs.beg);

                        std::vector<char> buffer(file_length);

                        while (!glyph_ifs.eof())
                        {
                            glyph_ifs.read(buffer.data(), 1024);
                        }

                        std::string content(buffer.begin(), buffer.end());

                        while (!content.empty())
                        {
                            const auto pos = content.find_first_of("\n\r");

                            std::string current_line;

                            if (pos != std::string::npos)
                            {
                                current_line = content.substr(0, pos);
                                content = content.substr(pos + 1);
                            }
                            else
                            {
                                current_line = content;
                                content = "";
                            }

                            if (!current_line.empty() && current_line.find_first_of('#') == std::string::npos)
                            {
                                const auto num_values = std::count(current_line.begin(), current_line.end(), ',') + 1;

                                if (num_values > 0 && num_values % 2 == 0)
                                {
                                    auto num_points = num_values / 2;

                                    if (num_points == 1)
                                    {
                                        const auto first_value_str = current_line.substr(0, current_line.find_first_of(','));
                                        const auto second_value_str = current_line.substr(current_line.find_first_of(',') + 1);

                                        const float first_value = std::stof(first_value_str);
                                        const float second_value = std::stof(second_value_str);

                                        this->stored_data.points.push_back(std::make_pair(static_cast<float>(id++), Eigen::Vector2f(first_value, second_value)));

                                        this->stored_data.hash = core::utility::DataHash(this->stored_data.hash, first_value, second_value);
                                    }
                                    else
                                    {
                                        std::vector<Eigen::Vector2f> line;

                                        while (num_points > 0)
                                        {
                                            const auto first_value_str = current_line.substr(0, current_line.find_first_of(','));
                                            const auto second_value_str = current_line.substr(current_line.find_first_of(',') + 1);

                                            const auto pos = current_line.find_first_of(',', current_line.find_first_of(',') + 1);

                                            current_line = current_line.substr(pos + 1);

                                            const float first_value = std::stof(first_value_str);
                                            const float second_value = std::stof(second_value_str);

                                            line.push_back(Eigen::Vector2f(first_value, second_value));

                                            --num_points;
                                        }

                                        this->stored_data.lines.push_back(std::make_pair(static_cast<float>(id++), line));

                                        for (const auto& point : line)
                                        {
                                            this->stored_data.hash = core::utility::DataHash(this->stored_data.hash, point[0], point[1]);
                                        }
                                    }
                                }
                                else
                                {
                                    vislib::sys::Log::DefaultLog.WriteWarn("Illegal glyph file '%s'!",
                                        this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                                    return false;
                                }
                            }
                        }
                    }
                    else
                    {
                        vislib::sys::Log::DefaultLog.WriteWarn("Unable to open input glyph file '%s'!",
                            this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }
                }
                
                if (this->stored_data.hash != glyph_call->DataHash())
                {
                    glyph_call->clear();

                    for (const auto& point : this->stored_data.points)
                    {
                        glyph_call->add_point(point.second, point.first);
                    }

                    for (const auto& line : this->stored_data.lines)
                    {
                        glyph_call->add_line(line.second, line.first);
                    }

                    glyph_call->SetDataHash(this->stored_data.hash);
                }
            }
            else
            {
                return false;
            }

            return true;
        }
    }
}
