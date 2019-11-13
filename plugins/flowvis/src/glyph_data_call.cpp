#include "stdafx.h"
#include "glyph_data_call.h"

#include "vislib/math/Rectangle.h"

#include "Eigen/Dense"

#include <memory>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        glyph_data_call::glyph_data_call() : bounding_rectangle_valid(false)
        {
            this->point_vertices = std::make_shared<std::vector<float>>();
            this->line_vertices = std::make_shared<std::vector<float>>();

            this->point_indices = std::make_shared<std::vector<unsigned int>>();
            this->line_indices = std::make_shared<std::vector<unsigned int>>();

            this->point_values = std::make_shared<std::vector<float>>();
            this->line_values = std::make_shared<std::vector<float>>();

            this->SetDataHash(-1);
        }

        const vislib::math::Rectangle<float>& glyph_data_call::get_bounding_rectangle() const
        {
            return this->bounding_rectangle;
        }

        void glyph_data_call::set_bounding_rectangle(const vislib::math::Rectangle<float>& bounding_rectangle)
        {
            this->bounding_rectangle = bounding_rectangle;
        }

        bool glyph_data_call::has_bounding_rectangle() const
        {
            return this->bounding_rectangle_valid;
        }

        std::shared_ptr<std::vector<float>> glyph_data_call::get_point_vertices() const
        {
            return this->point_vertices;
        }

        std::shared_ptr<std::vector<float>> glyph_data_call::get_line_vertices() const
        {
            return this->line_vertices;
        }

        std::shared_ptr<std::vector<unsigned int>> glyph_data_call::get_point_indices() const
        {
            return this->point_indices;
        }

        std::shared_ptr<std::vector<unsigned int>> glyph_data_call::get_line_indices() const
        {
            return this->line_indices;
        }

        std::shared_ptr<std::vector<float>> glyph_data_call::get_point_values() const
        {
            return this->point_values;
        }

        std::shared_ptr<std::vector<float>> glyph_data_call::get_line_values() const
        {
            return this->line_values;
        }

        void glyph_data_call::add_point(const Eigen::Vector2f& point, float value)
        {
            this->point_indices->push_back(static_cast<unsigned int>(this->point_vertices->size() / 2));

            this->point_vertices->push_back(point[0]);
            this->point_vertices->push_back(point[1]);

            this->point_values->push_back(value);

            // Adjust bounding rectangle
            if (this->bounding_rectangle_valid)
            {
                this->bounding_rectangle.SetLeft(std::min(this->bounding_rectangle.Left(), point[0]));
                this->bounding_rectangle.SetRight(std::max(this->bounding_rectangle.Right(), point[0]));

                this->bounding_rectangle.SetBottom(std::min(this->bounding_rectangle.Bottom(), point[1]));
                this->bounding_rectangle.SetTop(std::max(this->bounding_rectangle.Top(), point[1]));
            }
            else
            {
                this->bounding_rectangle.SetLeft(point[0]);
                this->bounding_rectangle.SetRight(point[0]);

                this->bounding_rectangle.SetBottom(point[1]);
                this->bounding_rectangle.SetTop(point[1]);
            }

            this->bounding_rectangle_valid = true;
        }

        void glyph_data_call::add_line(const std::vector<Eigen::Vector2f>& points, float value)
        {
            unsigned int next_index = static_cast<unsigned int>(this->line_vertices->size() / 2);

            // Push restart index
            if (next_index != 0) {
                this->line_indices->push_back(-1);
            }

            float min_x, min_y, max_x, max_y;
            min_x = max_x = points[0][0];
            min_y = max_y = points[0][1];

            // Add indices for a line strip
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                const auto& point = points[i];

                this->line_indices->push_back(next_index++);

                this->line_vertices->push_back(point[0]);
                this->line_vertices->push_back(point[1]);

                min_x = std::min(min_x, point[0]);
                max_x = std::max(max_x, point[0]);

                min_y = std::min(min_y, point[1]);
                max_y = std::max(max_y, point[1]);

                this->line_values->push_back(value);
            }

            // Adjust bounding rectangle
            if (this->bounding_rectangle_valid)
            {
                this->bounding_rectangle.SetLeft(std::min(this->bounding_rectangle.Left(), min_x));
                this->bounding_rectangle.SetRight(std::max(this->bounding_rectangle.Right(), max_x));

                this->bounding_rectangle.SetBottom(std::min(this->bounding_rectangle.Bottom(), min_y));
                this->bounding_rectangle.SetTop(std::max(this->bounding_rectangle.Top(), max_y));
            }
            else
            {
                this->bounding_rectangle.SetLeft(min_x);
                this->bounding_rectangle.SetRight(max_x);

                this->bounding_rectangle.SetBottom(min_y);
                this->bounding_rectangle.SetTop(max_y);
            }

            this->bounding_rectangle_valid = true;
        }

        std::vector<std::pair<Eigen::Vector2f, float>> glyph_data_call::get_points() const {
            std::vector<std::pair<Eigen::Vector2f, float>> points(this->point_indices->size());

            for (std::size_t i = 0; i < points.size(); ++i) {
                const auto index = static_cast<std::size_t>((*this->point_indices)[i]);

                points[i].first << (*this->point_vertices)[index * 2], (*this->point_vertices)[index * 2 + 1];
                points[i].second = (*this->point_values)[index];
            }

            return points;
        }

        std::vector<std::pair<std::vector<Eigen::Vector2f>, float>> glyph_data_call::get_lines() const {
            std::vector<std::pair<std::vector<Eigen::Vector2f>, float>> lines;

            if (!this->line_indices->empty()) {
                lines.emplace_back();
                lines.back().second = (*this->line_values)[(*this->line_indices)[0]];

                for (std::size_t i = 0; i < this->line_indices->size(); ++i) {
                    const auto index = (*this->line_indices)[i];

                    if (index == -1) {
                        // Restart index found: add new line
                        lines.emplace_back();
                        lines.back().second = (*this->line_values)[(*this->line_indices)[i + 1]];
                    } else {
                        lines.back().first.emplace_back(
                            (*this->line_vertices)[static_cast<std::size_t>(index) * 2],
                            (*this->line_vertices)[static_cast<std::size_t>(index) * 2 + 1]);
                    }
                }
            }

            return lines;
        }

        std::vector<std::pair<std::pair<Eigen::Vector2f, Eigen::Vector2f>, float>> glyph_data_call::get_line_segments() const {
            std::vector<std::pair<std::pair<Eigen::Vector2f, Eigen::Vector2f>, float>> line_segments;

            if (!this->line_indices->empty()) {
                const auto lines = get_lines();

                for (const auto& line : lines) {
                    for (std::size_t segment = 0; segment < line.first.size() - 1; ++segment) {
                        line_segments.push_back(
                            std::make_pair(std::make_pair(line.first[segment], line.first[segment + 1]), line.second));
                    }
                }
            }

            return line_segments;
        }

        void glyph_data_call::clear()
        {
            this->point_vertices->clear();
            this->line_vertices->clear();

            this->point_indices->clear();
            this->line_indices->clear();

            this->point_values->clear();
            this->line_values->clear();

            this->bounding_rectangle_valid = false;
        }
    }
}