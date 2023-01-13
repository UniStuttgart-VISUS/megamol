/*
 * BindingSiteData.h
 *
 * Author: Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "glm/glm.hpp"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace protein {

class BindingSiteDataSource : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "BindingSiteDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers binding site information for biomolecules.";
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
    BindingSiteDataSource(void);

    /** dtor */
    ~BindingSiteDataSource(void);

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

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(megamol::core::Call& call);

private:
    /**
     * Load information about amino acids and residues from a PDB file.
     *
     * @param filename The PDB file name.
     */
    void loadPDBFile(const std::string& filename);

    /**
     * TODO
     */
    std::string ExtractBindingSiteDescripton(std::string bsName, std::vector<std::string> remarkArray);

    /** The data callee slot */
    core::CalleeSlot dataOutSlot_;

    /** the parameter slot for the binding site file (PDB) */
    core::param::ParamSlot pdbFilenameSlot_;
    // the file name for the color table
    megamol::core::param::ParamSlot colorTableFileParam_;

    /** Parameter to activate the special enzyme mode */
    megamol::core::param::ParamSlot enzymeModeParam_;

    /** Parameter to select the type of the protein */
    megamol::core::param::ParamSlot gxTypeFlag_;

    /** The binding site information */
    std::vector<std::vector<std::pair<char, unsigned int>>> bindingSites_;
    /** Pointer to binding site residue name array */
    std::vector<std::vector<std::string>> bindingSiteResNames_;
    /** The binding site name */
    std::vector<std::string> bindingSiteNames_;
    /** The binding site description */
    std::vector<std::string> bindingSiteDescription_;

    // color table
    std::vector<glm::vec3> colorLookupTable_;
    // color table
    std::vector<glm::vec3> bindingSiteColors_;
};

} /* end namespace protein */
} /* end namespace megamol */
