#include "stdafx.h"
#include "vector_field_reader.h"

#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/param/FilePathParam.h"

#include "vislib/sys/Log.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        vector_field_reader::vector_field_reader() :
            output_slot("output_slot", "Output slot for the vector field"),
            file_path_slot("file_path_slot", "File path to the stored vector field")
        {
            // Set connections and parameters
            this->output_slot.SetCallback(vector_field_call::ClassName(), vector_field_call::FunctionName(0), &vector_field_reader::get_data);
            this->output_slot.SetCallback(vector_field_call::ClassName(), vector_field_call::FunctionName(1), &vector_field_reader::get_extent);
            this->MakeSlotAvailable(&this->output_slot);

            this->file_path_slot << new core::param::FilePathParam("");
            this->MakeSlotAvailable(&this->file_path_slot);

            // Initialize stored data
            this->stored_data.bounding_rectangle = vislib::math::Rectangle<float>(0.0f, 0.0f, 1.0f, 1.0f);
            this->stored_data.resolution = { 0u, 0u };
            this->stored_data.positions = std::make_shared<std::vector<float>>();
            this->stored_data.vectors = std::make_shared<std::vector<float>>();
        }

        vector_field_reader::~vector_field_reader()
        {
            this->Release();
        }

        bool vector_field_reader::create()
        {
            return true;
        }

        void vector_field_reader::release()
        { }

        bool vector_field_reader::get_data(core::Call& call)
        {
            // Get call
            auto* vf_call = dynamic_cast<vector_field_call*>(&call);

            if (vf_call != nullptr)
            {
                if (!this->file_path_slot.Param<core::param::FilePathParam>()->Value().IsEmpty() &&
                    this->file_path_slot.IsDirty())
                {
                    this->file_path_slot.ResetDirty();

                    // Open file
                    std::ifstream vectors_ifs(this->file_path_slot.Param<core::param::FilePathParam>()->Value(), std::ios_base::in | std::ios_base::binary);

                    if (vectors_ifs.good())
                    {
                        // Get dimension from file
                        unsigned int dimension, components;

                        vectors_ifs.read(reinterpret_cast<char*>(&dimension), sizeof(unsigned int));
                        vectors_ifs.read(reinterpret_cast<char*>(&components), sizeof(unsigned int));

                        if (dimension != 2)
                        {
                            vislib::sys::Log::DefaultLog.WriteError("Vector field file must have exactly two dimensions '%s'",
                                this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                            return false;
                        }

                        if (components != 2)
                        {
                            vislib::sys::Log::DefaultLog.WriteError("Vectors must have exactly two components '%s'",
                                this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                            return false;
                        }

                        // Read extents from file
                        float x_min, x_max, y_min, y_max;
                        unsigned int x_num, y_num, num;

                        vectors_ifs.read(reinterpret_cast<char*>(&x_num), sizeof(unsigned int));
                        vectors_ifs.read(reinterpret_cast<char*>(&x_min), sizeof(float));
                        vectors_ifs.read(reinterpret_cast<char*>(&x_max), sizeof(float));
                        vectors_ifs.read(reinterpret_cast<char*>(&y_num), sizeof(unsigned int));
                        vectors_ifs.read(reinterpret_cast<char*>(&y_min), sizeof(float));
                        vectors_ifs.read(reinterpret_cast<char*>(&y_max), sizeof(float));

                        num = x_num * y_num;

                        vf_call->set_resolution(this->stored_data.resolution = { static_cast<unsigned int>(x_num), static_cast<unsigned int>(y_num) });
                        vf_call->set_bounding_rectangle(this->stored_data.bounding_rectangle = vislib::math::Rectangle<float>(x_min, y_min, x_max, y_max));

                        // Read file content
                        const float x_step = (x_max - x_min) / (x_num - 1);
                        const float y_step = (y_max - y_min) / (y_num - 1);

                        auto positions = std::make_shared<std::vector<float>>(num * 2);
                        auto vectors = std::make_shared<std::vector<float>>(num * 2);

                        for (unsigned int y = 0; y < y_num; ++y)
                        {
                            for (unsigned int x = 0; x < x_num; ++x)
                            {
                                const unsigned int xy = y * x_num + x;

                                // Calculate positions
                                const float x_pos = x_min + x * x_step;
                                const float y_pos = y_min + y * y_step;

                                (*positions)[xy * 2 + 0] = x_pos;
                                (*positions)[xy * 2 + 1] = y_pos;

                                // Read vectors
                                vectors_ifs.read(reinterpret_cast<char*>(&(*vectors)[xy * 2 + 0]), sizeof(float));
                                vectors_ifs.read(reinterpret_cast<char*>(&(*vectors)[xy * 2 + 1]), sizeof(float));
                            }
                        }

                        vf_call->set_positions(this->stored_data.positions = positions);
                        vf_call->set_vectors(this->stored_data.vectors = vectors);

                        vectors_ifs.close();

                        vf_call->SetDataHash(this->stored_data.hash = (vf_call->DataHash() + 1));

                        return true;
                    }
                    else
                    {
                        vislib::sys::Log::DefaultLog.WriteWarn("Unable to open input vector field file '%s'!",
                            this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }
                }
                else
                {
                    vf_call->set_resolution(this->stored_data.resolution);
                    vf_call->set_bounding_rectangle(this->stored_data.bounding_rectangle);

                    vf_call->set_positions(this->stored_data.positions);
                    vf_call->set_vectors(this->stored_data.vectors);

                    vf_call->SetDataHash(this->stored_data.hash);

                    return true;
                }
            }
            else
            {
                return false;
            }

            return true;
        }

        bool vector_field_reader::get_extent(core::Call& call)
        {
            // Get call
            auto* vf_call = dynamic_cast<vector_field_call*>(&call);

            if (vf_call != nullptr)
            {
                if (!this->file_path_slot.Param<core::param::FilePathParam>()->Value().IsEmpty() && this->file_path_slot.IsDirty())
                {
                    // Open file
                    std::ifstream vectors_ifs(this->file_path_slot.Param<core::param::FilePathParam>()->Value(), std::ios_base::in | std::ios_base::binary);

                    if (vectors_ifs.good())
                    {
                        // Get dimension from file
                        unsigned int dimension, components;

                        vectors_ifs.read(reinterpret_cast<char*>(&dimension), sizeof(unsigned int));
                        vectors_ifs.read(reinterpret_cast<char*>(&components), sizeof(unsigned int));

                        if (dimension != 2)
                        {
                            vislib::sys::Log::DefaultLog.WriteError("Vector field file must have exactly two dimensions '%s'",
                                this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                            return false;
                        }

                        if (components != 2)
                        {
                            vislib::sys::Log::DefaultLog.WriteError("Vectors must have exactly two components '%s'",
                                this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                            return false;
                        }

                        // Read extents from file
                        float x_min, x_max, y_min, y_max;
                        unsigned int x_num, y_num, num;

                        vectors_ifs.read(reinterpret_cast<char*>(&x_num), sizeof(unsigned int));
                        vectors_ifs.read(reinterpret_cast<char*>(&x_min), sizeof(float));
                        vectors_ifs.read(reinterpret_cast<char*>(&x_max), sizeof(float));
                        vectors_ifs.read(reinterpret_cast<char*>(&y_num), sizeof(unsigned int));
                        vectors_ifs.read(reinterpret_cast<char*>(&y_min), sizeof(float));
                        vectors_ifs.read(reinterpret_cast<char*>(&y_max), sizeof(float));

                        num = x_num * y_num;

                        vf_call->set_resolution(this->stored_data.resolution = { static_cast<unsigned int>(x_num), static_cast<unsigned int>(y_num) });
                        vf_call->set_bounding_rectangle(this->stored_data.bounding_rectangle = vislib::math::Rectangle<float>(x_min, y_min, x_max, y_max));

                        vectors_ifs.close();

                        return true;
                    }
                    else
                    {
                        vislib::sys::Log::DefaultLog.WriteWarn("Unable to open input vector field file '%s'!",
                            this->file_path_slot.Param<core::param::FilePathParam>()->Value());

                        return false;
                    }
                }
                else
                {
                    vf_call->set_resolution(this->stored_data.resolution);
                    vf_call->set_bounding_rectangle(this->stored_data.bounding_rectangle);

                    return true;
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
