#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "datatools/clustering/ann_interface.h"

#include "datatools/clustering/DBSCAN2.h"

#include "datatools/table/TableDataCall.h"

namespace megamol {
namespace adios {

class Clustering : public core::Module {
public:
    enum Algorithm { DBSCAN };

    /** Return module class name */
    static const char* ClassName(void) {
        return "Clustering";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Applies clustering on input";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    Clustering(void);

    /** Dtor */
    virtual ~Clustering(void);

protected:
    /** Lazy initialization of the module */
    bool create(void) override;

    /** Resource release */
    void release(void) override;

private:
    bool getDataCallback(core::Call& c);
    bool getHeaderCallback(core::Call& c);

    bool changeAlgCallback(core::param::ParamSlot& p);

    void fillDataVec(
        datatools::clustering::clusters_t const& clusters, float const* table, size_t num_rows, size_t num_columns) {
        data_.resize(num_rows * (num_columns + 1));
        for (size_t row = 0; row < num_rows; ++row) {
            data_[row * (num_columns + 1)] = clusters[row];
            for (size_t col = 0; col < num_columns; ++col) {
                data_[(col + 1) + row * (num_columns + 1)] = table[col + row * num_columns];
            }
        }
    }

    bool isDirty() {
        return min_pts_slot_.IsDirty() || sigma_slot_.IsDirty();
    }

    void resetDirty() {
        min_pts_slot_.ResetDirty();
        sigma_slot_.ResetDirty();
    }

    core::CalleeSlot data_out_slot_;
    core::CallerSlot data_in_slot_;

    core::param::ParamSlot alg_selector_slot_;

    core::param::ParamSlot min_pts_slot_;
    core::param::ParamSlot sigma_slot_;

    std::vector<float> data_;
    std::vector<datatools::table::TableDataCall::ColumnInfo> infos_;
    size_t out_num_rows_;
    size_t out_num_cols_;

    size_t data_hash_;
}; // class Clustering

} // namespace adios
} // namespace megamol
