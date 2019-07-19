/*
 * FEMTxtLoader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef FEM_TXT_LOADER_H_INCLUDED
#define FEM_TXT_LOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "FEMDataCall.h"
#include "archvis/archvis.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace archvis {

class FEMLoader : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FEMLoader"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for simply loading txt-based FEM files from disk"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    FEMLoader();
    ~FEMLoader();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

    std::vector<FEMModel::Vec3> loadNodesFromFile(std::string const& filename);

    std::vector<std::array<size_t, 8>> loadElementsFromFile(std::string const& filename);

    std::vector<FEMModel::Vec4> loadNodeDeformationsFromFile(std::string const& filename);

private:
    std::shared_ptr<FEMModel> m_fem_data;

    int m_update_flag;

    /** The fem node file name */
    core::param::ParamSlot m_femNodes_filename_slot;

    /** The fem elements file name */
    core::param::ParamSlot m_femElements_filename_slot;

    /** Example displacement data */
    core::param::ParamSlot m_femDeformation_filename_slot;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_getData_slot;
};

} // namespace archvis
} // namespace megamol

#endif // !FEM_TXT_LOADER_H_INCLUDED
