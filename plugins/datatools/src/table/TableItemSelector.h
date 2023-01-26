#pragma once

#include <unordered_map>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"

namespace megamol {
namespace datatools {
namespace table {

class TableItemSelector : public core::Module {
public:
    enum class SELECT_ALG { FIRST_OF_ALL, MIN_DIFF };

    /** Return module class name */
    static const char* ClassName() {
        return "TableItemSelector";
    }

    /** Return module class description */
    static const char* Description() {
        return "Extracts most significant item in a selection. Selection defined by a column.";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    TableItemSelector();

    /** Dtor */
    ~TableItemSelector() override;

protected:
    /** Lazy initialization of the module */
    bool create() override;

    /** Resource release */
    void release() override;

private:
    bool getDataCallback(core::Call& c);
    bool getHashCallback(core::Call& c);

    void fill_enum(TableDataCall::ColumnInfo const* infos, size_t num_colums) {
        column_map_.clear();
        auto selection_column_param = selection_column_slot_.Param<core::param::FlexEnumParam>();

        for (size_t col = 0; col < num_colums; ++col) {
            column_map_[infos[col].Name()] = col;
            selection_column_param->AddValue(infos[col].Name());
        }
    }

    std::vector<float> extract_column(float const* data, size_t num_columns, size_t num_rows, size_t idx) {
        std::vector<float> ret(num_rows);

        for (size_t row = 0; row < num_rows; ++row) {
            ret[row] = data[idx + row * num_columns];
        }

        return ret;
    }

    bool is_dirty() {
        return selection_column_slot_.IsDirty();
    }

    void reset_dirty() {
        selection_column_slot_.ResetDirty();
    }

    std::vector<size_t> first_of_all(float const* data, size_t num_columns, size_t num_rows,
        std::vector<float> const& selector, size_t selector_idx) {
        std::unordered_map<float, size_t> cat;

        for (size_t row = 0; row < num_rows; ++row) {
            auto const val = selector[row];
            auto const fit = cat.find(val);
            if (fit == cat.end()) {
                cat[val] = row;
            }
        }

        std::vector<size_t> ret;
        ret.reserve(cat.size());

        for (auto const& el : cat) {
            ret.push_back(el.second);
        }

        return ret;
    }
    std::vector<size_t> min_difference(float const* data, size_t num_columns, size_t num_rows,
        std::vector<float> const& selector, size_t selector_idx) {
        std::vector<float> cat;

        for (auto const& el : selector) {
            auto const fit = std::find(cat.cbegin(), cat.cend(), el);
            if (fit == cat.cend()) {
                cat.push_back(el);
            }
        }

        std::vector<float> base(cat.size(), 0.0f);
        std::vector<size_t> counters(cat.size(), 0);
        std::vector<float> scores(num_rows, 0.0f);


        for (size_t row = 0; row < num_rows; ++row) {
            float avg = 0.0f;
            for (size_t col = 0; col < num_columns; ++col) {
                avg += data[col + row * num_columns];
            }
            avg /= static_cast<float>(num_columns);
            scores[row] = avg;

            auto const fit = std::find(cat.cbegin(), cat.cend(), selector[row]);
            if (fit != cat.cend()) {
                auto const cat_idx = std::distance(cat.cbegin(), fit);
                base[cat_idx] += avg;
                ++counters[cat_idx];
            }
        }
        for (size_t cat_idx = 0; cat_idx < cat.size(); ++cat_idx) {
            base[cat_idx] /= static_cast<float>(counters[cat_idx]);
        }

        std::vector<size_t> ret(cat.size());
        std::vector<float> diff(cat.size(), std::numeric_limits<float>::max());
        for (size_t row = 0; row < num_rows; ++row) {
            auto const fit = std::find(cat.cbegin(), cat.cend(), selector[row]);
            if (fit != cat.cend()) {
                auto const cat_idx = std::distance(cat.cbegin(), fit);
                auto const d = std::abs(base[cat_idx] - scores[row]);
                if (d < diff[cat_idx]) {
                    diff[cat_idx] = d;
                    ret[cat_idx] = row;
                }
            }
        }

        return ret;
    }

    std::vector<float> extract_rows(float const* data, size_t num_cols, size_t num_rows, std::vector<size_t> indices) {
        auto const num_peaks = indices.size();
        std::vector<float> ret(num_cols * num_peaks);

        for (size_t i = 0; i < num_peaks; ++i) {
            auto const idx = indices[i];
            std::copy(data + (idx * num_cols), data + ((idx + 1) * num_cols), ret.begin() + (i * num_cols));
        }

        return ret;
    }

    core::CalleeSlot data_out_slot_;
    core::CallerSlot data_in_slot_;

    core::param::ParamSlot selection_column_slot_;
    core::param::ParamSlot selection_algorithm_slot_;

    std::unordered_map<std::string, size_t> column_map_;

    std::vector<float> data_;
    std::vector<datatools::table::TableDataCall::ColumnInfo> infos_;
    size_t out_num_columns_;
    size_t out_num_rows_;

    size_t data_hash_;
};

} // namespace table
} // namespace datatools
} // namespace megamol
