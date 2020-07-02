/*
 * Module.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MODULE_H_INCLUDED
#define MEGAMOLCORE_MODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <string>
#include <vector>
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/api/MegaMolCore.std.h"

#include "RenderResource.h"


namespace megamol {
namespace core {

/** forward declaration */
class CoreInstance;
class AbstractSlot;
namespace factories {
class ModuleDescription;
}
class Module;

} /* end namespace core */
} /* end namespace megamol */

namespace std {

// dll-export of std-type instantiations
MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API shared_ptr<::megamol::core::Module>;

} // namespace std

namespace megamol {
namespace core {

/**
 * Base class of all graph modules
 */
class MEGAMOLCORE_API Module : public AbstractNamedObjectContainer {
public:
    virtual std::vector<std::string> requested_lifetime_dependencies() { 
		return {"IOpenGL_Context"}; 
	}

    friend class ::megamol::core::factories::ModuleDescription;

    /** Shared ptr type alias */
    typedef ::std::shared_ptr<Module> ptr_type;

    /** Shared ptr type alias */
    typedef ::std::shared_ptr<const Module> const_ptr_type;

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template <class T> inline static ptr_type dynamic_pointer_cast(std::shared_ptr<T> p) {
        return std::dynamic_pointer_cast<Module, T>(p);
    }

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template <class T> inline static const_ptr_type dynamic_pointer_cast(std::shared_ptr<const T> p) {
        return std::dynamic_pointer_cast<const Module, const T>(p);
    }

    /**
     * Answer whether or not this module supports being used in a
     * quickstart. Overwrite if you don't want your module to be used in
     * quickstarts.
     *
     * This default implementation returns 'true'
     *
     * @return Whether or not this module supports being used in a
     *         quickstart.
     */
    static bool SupportQuickstart(void) { return true; }

    /**
     * Ctor.
     *
     * Be aware of the fact that most of your initialisation code should
     * be placed in 'create' since the ctor cannot fail and important
     * members (such as 'instance') are set after the ctor returns.
     */
    Module(void);

    /** Dtor. */
    virtual ~Module(void);

    /**
     * Tries to create this module. Do not overwrite this method!
     * Overwrite 'create'!
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool Create(std::vector<megamol::render_api::RenderResource> dependencies = {});

    /**
     * Finds the slot with the given name.
     *
     * @param name The name of the slot to find (case sensitive!)
     *
     * @return The found slot of this module, or 'NULL' if there is no
     *         slot with this name.
     */
    AbstractSlot* FindSlot(const vislib::StringA& name);

    /**
     * Gets the name of the current instance as specified on the command
     * line, i.e. the name one level below the root namespace.
     * Caution: This can only work after the module is properly
     * inserted into the module graph and its parent is known,
     * calling it on create, e.g. will not yield satisfactory results.
     *
     * @return the according name
     */
    vislib::StringA GetDemiRootName() const;

    /**
     * Releases the module and all resources. Do not overwrite this method!
     * Overwrite 'release'!
     */
    void Release(std::vector<megamol::render_api::RenderResource> dependencies = {});

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark(void);

    /**
     * Performs the cleanup operation by removing and deleteing of all
     * marked objects.
     */
    virtual void PerformCleanup(void);

    inline void SetClassName(const char* name) { this->className = name; }

    inline const char* ClassName() const { return this->className; }

    bool isCreated() const { return this->created; }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) = 0;

    /**
     * Check the configuration for a value for the parameter 'val'.
     * It checks, in descending order of priority, for occurences
     * of: [this.GetDemiRootName]-[name], *-[name], [name] and
     * returns the respective value. If nothing is found,
     * vislib::StringA::EMPTY is returned.
     * Caution: This can only work after the module is properly
     * inserted into the module graph, since otherwise the
     * DemiRootName cannot be determined reliably
     *
     * @param name the name of the sought value
     *
     * @return the value or vislib::StringA::EMPTY
     */
    vislib::StringA getRelevantConfigValue(vislib::StringA name);

    /**
     * Gets the instance of the core owning this module.
     *
     * @return The instance of the core owning this module.
     */
    inline class ::megamol::core::CoreInstance* instance(void) const { return this->GetCoreInstance(); }

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void) = 0;

    /**
     * Makes the given slot for this module available.
     *
     * @param slot Slot to be made available.
     */
    void MakeSlotAvailable(AbstractSlot* slot);
    void SetSlotUnavailable(AbstractSlot* slot);

private:
    /** Sets the name of the module */
    void setModuleName(const vislib::StringA& name);

    /** Flag whether this module is created or not */
    bool created;

    const char* className;

    /* Allow the container to access the internal create flag */
    friend class ::megamol::core::AbstractNamedObjectContainer;
};


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MODULE_H_INCLUDED */
