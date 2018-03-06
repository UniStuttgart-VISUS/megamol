/*
 * PBSDataSource.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef PBS_PBSREADER_H_INCLUDED
#define PBS_PBSREADER_H_INCLUDED

#include <array>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"

#include "zfp.h"

#include "PBSStorage.h"

namespace megamol {
namespace pbs {

/** Module to read ZFP compressed point dumps. */
class PBSDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "PBSDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Data reader module for ZFP compressed point dumps files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    PBSDataSource(void);

    /** dtor */
    virtual ~PBSDataSource(void);
protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);
private:
    /** Idx enum for point attributes */
    enum attribute_type {
        x = 0,
        y,
        z,
        nx,
        ny,
        cr,
        cg,
        cb
    };

    /*enum pbs_type {
        None,
        P,
        PN,
        PC,
        PNC
    };*/

    /**
     * Helper function to insert read buffer into an output buffer.
     *
     * @lhs Buffer to insert into
     * @rhs Buffer to insert from
     * @num_elements Number of elements to insert
     */
    template<class T>
    inline void insertElements(std::shared_ptr<std::vector<T>>& lhs, std::vector<char>& rhs, size_t num_elements) {
        if (num_elements > rhs.size() / sizeof(T)) {
            throw std::out_of_range("PBSDataSource::insertElements: rhs.size() to small for num_elements\n");
        }
        //lhs->insert(lhs->end(), reinterpret_cast<T*>(rhs.data()), reinterpret_cast<T*>(rhs.data()) + num_elements * sizeof(T));
        auto idx = lhs->size();
        lhs->resize(lhs->size() + num_elements);
        /*for (size_t i = 0; i < num_elements * sizeof(T); i+=sizeof(T)) {
            lhs->push_back(*reinterpret_cast<T*>(&rhs[i]));
        }*/
        memcpy(&((lhs->data())[idx]), rhs.data(), num_elements * sizeof(T)); // <-- don't do this at home
    }

    /**
     * Clears all buffers.
     */
    void clearBuffers(void);

    /**
     * Reads and decompressed PBS file
     *
     * @param filename Path to the PBS file without extension
     * @param data Buffer to store the uncompressed file content
     *
     * @return True, if file was successfully read
     */
    bool readPBSFile(std::ifstream& file, const size_t file_buffer_size, std::vector<char> &data, std::vector<char> &tmp, const zfp_type type, const unsigned int num_elements, const double tol);

    bool read(void);

    bool isDirty(void);

    void resetDirty(void);

    /**
     * Callback receiving the update of the file name parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool filenameChanged(core::param::ParamSlot& slot);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& c);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& c);

    /** The file name */
    core::param::ParamSlot filenameSlot;

    /** The start chunk idx */
    core::param::ParamSlot start_idx_slot;

    /** The end chunk idx */
    core::param::ParamSlot end_idx_slot;

    /** The start render region idx */
    core::param::ParamSlot start_region_idx_slot;

    /** The end render region idx */
    core::param::ParamSlot end_region_idx_slot;

    /** The slot for requesting data */
    core::CalleeSlot getData;

    /** Buffer holding the data of the PBS file */
    std::vector<char> data;

    /** The data set data hash */
    size_t dataHash;

    /** Lookup table for ZFP datatype sizes */
    size_t datatype_size[5] = {
        0, sizeof(int32_t), sizeof(int64_t), sizeof(float), sizeof(double)
    };

    /** Buffers for point coordinates */
    std::shared_ptr<std::vector<PBSStorage::pbs_coord_t>> x_data, y_data, z_data;

    /** Buffers for point normals in spherical coordinates */
    std::shared_ptr<std::vector<PBSStorage::pbs_normal_t>> nx_data, ny_data;

    /** Buffers for point colors */
    std::shared_ptr<std::vector<PBSStorage::pbs_color_t>> cr_data, cg_data, cb_data;

    /** Flag-storage for "renderable" property */
    std::vector<bool> render_flag;

    /** Number of possible attributes */
    static const unsigned int max_num_attributes = 8;

    std::array<bool, max_num_attributes> type_flags;

    /** Prefixes of filenames */
    std::string filename_prefixes[max_num_attributes] = {
        "x", "y", "z", "nx", "ny", "cr", "cg", "cb"
    };

    bool with_normals;

    bool with_colors;

    std::shared_ptr<double> g_bbox;

    std::shared_ptr<double> l_bbox;
};

} /* end namespace pbs */
} /* end namespace megamol */

#endif // end ifndef PBS_PBSREADER_H_INCLUDED
