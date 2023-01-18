/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <typeinfo>

#include "ModuleDescription.h"
#include "mmcore/utility/log/Log.h"

namespace megamol::core::factories {

/**
 * Class of rendering graph module descriptions generated using static
 * member implementations of the module classes.
 * Template parameter 'C' is the module class to be described.
 *
 * 'C' must implement the static methods:
 *      const char* ClassName();
 *      const char* Description();
 *      bool IsAvailable();
 */
template<class C>
class ModuleAutoDescription : public ModuleDescription {
public:
    /** Ctor. */
    ModuleAutoDescription() : ModuleDescription() {}

    /** Dtor. */
    ~ModuleAutoDescription() override = default;

    /**
     * Answer the name of the module described.
     *
     * @return The name of the module described.
     */
    const char* ClassName() const override {
        return C::ClassName();
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    const char* Description() const override {
        return C::Description();
    }

    /**
     * Answers whether this module is available on the current system.
     * This implementation always returns 'true'.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    bool IsAvailable() const override {
        return C::IsAvailable();
    }

    void requested_lifetime_resources(frontend_resources::ResourceRequest& req) const override {
        return C::requested_lifetime_resources(req);
    }

    /**
     * Answers whether this description is describing the class of
     * 'module'.
     *
     * @param module The module to test.
     *
     * @return 'true' if 'module' is described by this description,
     *         'false' otherwise.
     */
    bool IsDescribing(const Module* module) const override {
        // return dynamic_cast<const C*>(module) != NULL;
        // mueller: The version above depends on modules being ordered from
        // most-specialised to base classes. This is cannot be guaranteed
        // and has the effect that the network serialisation uses wrong
        // types. The new RTTI version only allows exact matches. We believe
        // that this does not have any negative effects somewhere else in the
        // programme.
        return (typeid(C) == typeid(*module));
    }

    /**
     * Answer whether or not this module can be used in a quickstart
     *
     * @return 'true' if the module can be used in a quickstart
     */
    bool IsVisibleForQuickstart() const override {
        return C::SupportQuickstart();
    }

protected:
    /**
     * Creates a new module object from this description.
     *
     * @return The newly created module object or 'NULL' in case of an
     *         error.
     */
    Module::ptr_type createModuleImpl() const override {
        using megamol::core::utility::log::Log;
        try {
            Module::ptr_type m = std::make_shared<C>();
            m->SetClassName(this->ClassName());
            return m;
        } catch (vislib::Exception& ex) {
            Log::DefaultLog.WriteError("Exception while creating module %s: %s\n", C::ClassName(), ex.GetMsgA());
        } catch (std::exception& ex) {
            Log::DefaultLog.WriteError("Exception while creating module %s: %s\n", C::ClassName(), ex.what());
        } catch (...) {
            Log::DefaultLog.WriteError("Exception while creating module %s: Unknown exception\n", C::ClassName());
        }
        return nullptr;
    }
};

} // namespace megamol::core::factories
