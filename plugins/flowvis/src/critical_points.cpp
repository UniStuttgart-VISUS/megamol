#include "stdafx.h"
#include "critical_points.h"

#include "glyph_data_call.h"
#include "vector_field_call.h"

#include "mmcore/Call.h"
#include "mmcore/utility/DataHash.h"
#include "mmcore/param/IntParam.h"

#include "vislib/math/Rectangle.h"
#include "vislib/sys/Log.h"

#include "Eigen/Dense"

#include <array>

namespace megamol
{
    namespace flowvis
    {
        critical_points::critical_points() :
            glyph_slot("set_glyphs", "Glyph output"),
            glyph_hash(-1),
            vector_field_slot("get_vector_field", "Vector field input"),
            vector_field_hash(-1),
            boundary("boundary", "Number of boundary cells")
        {
            // Connect output
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(0), &critical_points::get_glyph_data_callback);
            this->glyph_slot.SetCallback(glyph_data_call::ClassName(), glyph_data_call::FunctionName(1), &critical_points::get_glyph_extent_callback);
            this->MakeSlotAvailable(&this->glyph_slot);

            // Connect input
            this->vector_field_slot.SetCompatibleCall<vector_field_call::vector_field_description>();
            this->MakeSlotAvailable(&this->vector_field_slot);

            // Create parameter
            this->boundary << new core::param::IntParam(0);
            this->MakeSlotAvailable(&this->boundary);
        }

        critical_points::~critical_points()
        {
            this->Release();
        }

        bool critical_points::create()
        {
            return true;
        }

        void critical_points::release()
        {
        }

        bool critical_points::get_glyph_data_callback(core::Call& call)
        {
            auto* glyph_call = dynamic_cast<glyph_data_call*>(&call);

            if (glyph_call != nullptr)
            {
                // Get vector field
                auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();

                if (get_vector_field != nullptr && (*get_vector_field)(0) && get_vector_field->DataHash() != this->vector_field_hash)
                {
                    this->vector_field_hash = get_vector_field->DataHash();

                    const auto& positions = *get_vector_field->get_positions();
                    const auto& vectors = *get_vector_field->get_vectors();

                    // Reset output
                    this->glyph_output.clear();
                    this->glyph_hash = -1;

                    // Extract critical points
                    bool has_unhandled_case = false;

                    const unsigned int boundary_layer = static_cast<unsigned int>(this->boundary.Param<core::param::IntParam>()->Value());

                    for (unsigned int y = boundary_layer; y < get_vector_field->get_resolution()[1] - 1 - boundary_layer; ++y)
                    {
                        for (unsigned int x = boundary_layer; x < get_vector_field->get_resolution()[0] - 1 - boundary_layer; ++x)
                        {
                            const auto index_bottom_left = x + y * get_vector_field->get_resolution()[0];
                            const auto index_bottom_right = x + 1 + y * get_vector_field->get_resolution()[0];
                            const auto index_top_left = x + (y + 1) * get_vector_field->get_resolution()[0];
                            const auto index_top_right = x + 1 + (y + 1) * get_vector_field->get_resolution()[0];

                            const cell_t cell = {
                                Eigen::Vector2f(vectors[index_bottom_left * 2 + 0], vectors[index_bottom_left * 2 + 1]),
                                Eigen::Vector2f(vectors[index_bottom_right * 2 + 0], vectors[index_bottom_right * 2 + 1]),
                                Eigen::Vector2f(vectors[index_top_left * 2 + 0], vectors[index_top_left * 2 + 1]),
                                Eigen::Vector2f(vectors[index_top_right * 2 + 0], vectors[index_top_right * 2 + 1]),
                                Eigen::Vector2f(positions[index_bottom_left * 2 + 0], positions[index_bottom_left * 2 + 1]),
                                Eigen::Vector2f(positions[index_top_right * 2 + 0], positions[index_top_right * 2 + 1])
                            };

                            const auto critical_point = extract_critical_point(cell);

                            if (critical_point.first != type::NONE && critical_point.first != type::UNHANDLED)
                            {
                                this->glyph_output.push_back(critical_point);

                                this->glyph_hash = static_cast<SIZE_T>(core::utility::DataHash(this->glyph_hash,
                                    critical_point.first, critical_point.second[0], critical_point.second[1]));
                            }
                            else if (critical_point.first == type::UNHANDLED)
                            {
                                has_unhandled_case = true;
                            }
                        }
                    }

                    if (has_unhandled_case)
                    {
                        vislib::sys::Log::DefaultLog.WriteWarn("Unhandled case while extracting critical points");
                    }
                }

                // Fill glyph call
                if (glyph_call->DataHash() != this->glyph_hash)
                {
                    glyph_call->clear();

                    for (std::size_t i = 0; i < this->glyph_output.size(); ++i)
                    {
                        glyph_call->add_point(this->glyph_output[i].second, static_cast<float>(this->glyph_output[i].first));
                    }

                    glyph_call->SetDataHash(this->glyph_hash);
                }
            }

            return true;
        }

        bool critical_points::get_glyph_extent_callback(core::Call& call)
        {
            auto* get_vector_field = this->vector_field_slot.CallAs<vector_field_call>();

            return get_vector_field != nullptr && (*get_vector_field)(1);
        }

        std::pair<critical_points::type, Eigen::Vector2f> critical_points::extract_critical_point(const cell_t& cell) const
        {
            // Return the point directly, if it is a zero-vector itself
            if (cell.bottom_left.isZero())
            {
                return std::make_pair(type::NONE, cell.bottom_left_corner);
            }

            // Calculate cell size
            const auto size = cell.top_right_corner - cell.bottom_left_corner;

            const Eigen::Vector2f top_left_corner = cell.bottom_left_corner + Eigen::Vector2f(0.0, size[1]);
            const Eigen::Vector2f bottom_right_corner = cell.bottom_left_corner + Eigen::Vector2f(size[0], 0.0);

            // Calculate marching squares index
            int marching_squares_index = 0;

            if (cell.bottom_left[0] < 0) marching_squares_index += 1;
            if (cell.bottom_right[0] < 0) marching_squares_index += 2;
            if (cell.top_right[0] < 0) marching_squares_index += 4;
            if (cell.top_left[0] < 0) marching_squares_index += 8;

            // Find intersection
            if (marching_squares_index == 0)
            {
                return std::make_pair(type::NONE, Eigen::Vector2f());
            }

            // Create helper functions
            auto left = [&]() { return linear_interpolate_position(cell.bottom_left_corner, top_left_corner, cell.bottom_left[0], cell.top_left[0]); };
            auto right = [&]() { return linear_interpolate_position(bottom_right_corner, cell.top_right_corner, cell.bottom_right[0], cell.top_right[0]); };
            auto bottom = [&]() { return linear_interpolate_position(cell.bottom_left_corner, bottom_right_corner, cell.bottom_left[0], cell.bottom_right[0]); };
            auto top = [&]() { return linear_interpolate_position(top_left_corner, cell.top_right_corner, cell.top_left[0], cell.top_right[0]); };

            auto left_value = [&](const Eigen::Vector2f& position) { return linear_interpolate_value(cell.bottom_left_corner[1], cell.bottom_left_corner[1] + size[1], cell.bottom_left[1], cell.top_left[1], position[1]); };
            auto right_value = [&](const Eigen::Vector2f& position) { return linear_interpolate_value(cell.bottom_left_corner[1], cell.bottom_left_corner[1] + size[1], cell.bottom_right[1], cell.top_right[1], position[1]); };
            auto bottom_value = [&](const Eigen::Vector2f& position) { return linear_interpolate_value(cell.bottom_left_corner[0], cell.bottom_left_corner[0] + size[0], cell.bottom_left[1], cell.bottom_right[1], position[0]); };
            auto top_value = [&](const Eigen::Vector2f& position) { return linear_interpolate_value(cell.bottom_left_corner[0], cell.bottom_left_corner[0] + size[0], cell.top_left[1], cell.top_right[1], position[0]); };

            // Create line segment
            Eigen::Vector2f first, second;
            float first_value, second_value;

            switch (marching_squares_index)
            {
            case 1:
            case 3:
            case 7:
            case 8:
            case 12:
            case 14:
                // Start left
                first = left();
                first_value = left_value(first);

                // Get end point
                switch (marching_squares_index)
                {
                case 1:
                case 14:
                    second = bottom();
                    second_value = bottom_value(second);
                    break;

                case 3:
                case 12:
                    second = right();
                    second_value = right_value(second);
                    break;

                case 7:
                case 8:
                    second = top();
                    second_value = top_value(second);
                    break;
                }

                break;
            case 2:
            case 4:
            case 11:
            case 13:
                // Start right
                first = right();
                first_value = right_value(first);

                // Get end point
                switch (marching_squares_index)
                {
                case 2:
                case 13:
                    second = bottom();
                    second_value = bottom_value(second);
                    break;

                case 4:
                case 11:
                    second = top();
                    second_value = top_value(second);
                    break;
                }

                break;
            case 6:
            case 9:
                // Start bottom
                first = bottom();
                first_value = bottom_value(first);

                second = top();
                second_value = top_value(second);

                break;
            case 5:
            case 10:
            default:
                return std::make_pair(type::UNHANDLED, Eigen::Vector2f());
            }

            // Interpolate linearly between line end points
            if (std::signbit(first_value) != std::signbit(second_value))
            {
                const Eigen::Vector2f critical_point = linear_interpolate_position(first, second, first_value, second_value);

                // Classify critical point
                std::vector<Eigen::Vector2f> points, vectors;

                points.push_back(critical_point);
                vectors.push_back(bilinear_interpolate_value(cell, critical_point));

                points.push_back(cell.bottom_left_corner);
                vectors.push_back(cell.bottom_left);

                points.push_back(bottom_right_corner);
                vectors.push_back(cell.bottom_right);

                points.push_back(top_left_corner);
                vectors.push_back(cell.top_left);

                points.push_back(cell.top_right_corner);
                vectors.push_back(cell.top_right);

                const auto jacobian = calculate_jacobian(points, vectors);
                const auto eigenvalues = calculate_eigenvalues(jacobian);

                type point_type = type::NONE;

                if (eigenvalues.imag().isZero())
                {
                    if (eigenvalues.real().isZero())
                    {
                        point_type = type::NONE;
                    }
                    if (std::signbit(eigenvalues[0].real()) != std::signbit(eigenvalues[1].real()))
                    {
                        point_type = type::SADDLE;
                    }
                    else if (eigenvalues[0].real() > 0.0 && eigenvalues[1].real() > 0.0)
                    {
                        point_type = type::REPELLING_NODE;
                    }
                    else if (eigenvalues[0].real() < 0.0 && eigenvalues[1].real() < 0.0)
                    {
                        point_type = type::ATTRACTING_NODE;
                    }
                }
                else
                {
                    if (eigenvalues.real().isZero())
                    {
                        point_type = type::CENTER;
                    }
                    else if (std::signbit(eigenvalues[0].real()) != std::signbit(eigenvalues[1].real()))
                    {
                        point_type = type::NONE;
                    }
                    else if (eigenvalues[0].real() > 0.0 && eigenvalues[1].real() > 0.0)
                    {
                        point_type = type::REPELLING_FOCUS;
                    }
                    else if (eigenvalues[0].real() < 0.0 && eigenvalues[1].real() < 0.0)
                    {
                        point_type = type::ATTRACTING_FOCUS;
                    }
                }

                return std::make_pair(point_type, critical_point);
            }

            return std::make_pair(type::NONE, Eigen::Vector2f());
        }

        Eigen::Vector2f critical_points::linear_interpolate_position(const Eigen::Vector2f& left, const Eigen::Vector2f& right, const float value_left, const float value_right) const
        {
            const auto lambda = value_left / (value_left - value_right);

            return left + lambda * (right - left);
        }

        float critical_points::linear_interpolate_value(const float left, const float right, const float value_left, const float value_right, const float position) const
        {
            const auto width = right - left;

            const auto left_part = (position - left) / width;
            const auto right_part = 1.0f - left_part;

            return right_part * value_left + left_part * value_right;
        }

        Eigen::Vector2f critical_points::linear_interpolate_value(float left, float right, const Eigen::Vector2f& value_left, const Eigen::Vector2f& value_right, float position) const
        {
            const auto width = right - left;

            const auto left_part = (position - left) / width;
            const auto right_part = 1.0f - left_part;

            return right_part * value_left + left_part * value_right;
        }

        critical_points::cell_t::value_type critical_points::bilinear_interpolate_value(const cell_t& cell, const Eigen::Vector2f& position) const
        {
            const auto bottom = linear_interpolate_value(cell.bottom_left_corner[0], cell.top_right_corner[0], cell.bottom_left, cell.bottom_right, position[0]);
            const auto top = linear_interpolate_value(cell.bottom_left_corner[0], cell.top_right_corner[0], cell.top_left, cell.top_right, position[0]);

            return linear_interpolate_value(cell.bottom_left_corner[1], cell.top_right_corner[1], bottom, top, position[1]);
        }

        Eigen::Vector2f critical_points::calculate_gradient(const std::vector<Eigen::Vector2f>& vertices, const std::vector<float>& values) const
        {
            // Create matrix A, weight matrix W and right-hand side b
            Eigen::Matrix<float, Eigen::Dynamic, 2> A;
            A.resize(vertices.size() - 1, Eigen::NoChange);

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> W;
            W.resize(vertices.size() - 1, vertices.size() - 1);
            W.setZero();

            Eigen::Matrix<float, Eigen::Dynamic, 1> b;
            b.resize(vertices.size() - 1, Eigen::NoChange);

            for (std::size_t j = 0; j < vertices.size() - 1; ++j)
            {
                for (std::size_t d = 0; d < 2; ++d)
                {
                    A(j, d) = vertices[j + 1][d] - vertices[0][d];
                }

                W(j, j) = 1.0f / (vertices[j + 1] - vertices[0]).squaredNorm();

                b(j) = values[j + 1] - values[0];
            }

            // Calculate weighted matrix and vectors
            Eigen::Matrix<float, Eigen::Dynamic, 2> X;
            X.resize(vertices.size() - 1, Eigen::NoChange);

            X = W * A;
            b = W * b;

            // Solve least squares for Ax = b
            Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 2>> svd(X, Eigen::ComputeFullU | Eigen::ComputeFullV);

            return svd.solve(b);
        }

        Eigen::Matrix<float, 2, 2> critical_points::calculate_jacobian(const std::vector<Eigen::Vector2f>& vertices, const std::vector<Eigen::Vector2f>& values) const
        {
            // Get values component-wise
            auto x = std::vector<float>(values.size());
            auto y = std::vector<float>(values.size());

            for (std::size_t i = 0; i < values.size(); ++i)
            {
                x[i] = values[i][0];
                y[i] = values[i][1];
            }

            // Solve least squares for Ax = b
            Eigen::Vector2f result[2];

            result[0] = calculate_gradient(vertices, x);
            result[1] = calculate_gradient(vertices, y);

            // Return jacobian
            Eigen::Matrix<float, 2, 2> jacobian;

            for (std::size_t i = 0; i < 2; i++)
            {
                for (std::size_t j = 0; j < 2; j++)
                {
                    jacobian(j, i) = result[j][i];
                }
            }

            return jacobian;
        }

        Eigen::Vector2cf critical_points::calculate_eigenvalues(const Eigen::Matrix<float, 2, 2>& matrix) const
        {
            Eigen::EigenSolver<Eigen::Matrix<float, 2, 2>> eigensolver(matrix, true);

            return eigensolver.eigenvalues();
        }
    }
}
