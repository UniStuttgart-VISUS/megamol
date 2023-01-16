/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <string>

#include "ObjectDescription.h"
#include "mmcore/Module.h"

namespace megamol::core::factories {

/**
 * Abstract base class of rendering graph module descriptions
 */
class ModuleDescription : public ObjectDescription {
public:
    typedef std::shared_ptr<const ModuleDescription> ptr;

    /** Ctor. */
    ModuleDescription() = default;

    /** Dtor. */
    ~ModuleDescription() override = default;

    /**
     * Answer the class name of the module described.
     *
     * @return The class name of the module described.
     */
    const char* ClassName() const override = 0;

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    const char* Description() const override = 0;

    /**
     * Answers whether this module is available on the current system.
     * This implementation always returns 'true'.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    virtual bool IsAvailable() const;

    /**
     * Answers whether this description is describing the class of
     * 'module'.
     *
     * @param module The module to test.
     *
     * @return 'true' if 'module' is described by this description,
     *         'false' otherwise.
     */
    virtual bool IsDescribing(const Module* module) const = 0;

    /**
     * Creates a new module object from this description.
     *
     * @param name The name for the module to be created.
     * @param instance The core instance calling. Must not be 'NULL'.
     *
     * @return The newly created module object or 'NULL' in case of an
     *         error.
     */
    Module::ptr_type CreateModule(const std::string& name) const;

protected:
    /**
     * Creates a new module object from this description.
     *
     * @return The newly created module object or 'NULL' in case of an
     *         error.
     */
    virtual Module::ptr_type createModuleImpl() const = 0;
};

} // namespace megamol::core::factories
