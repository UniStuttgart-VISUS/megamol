/*
 * ServiceManager.h
 *
 * Copyright (C) 2016 by MegaMol Team (S. Grottel)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_SERVICEMANAGER_H_INCLUDED
#define MEGAMOLCORE_SERVICEMANAGER_H_INCLUDED
#pragma once

#include "mmcore/AbstractService.h"
#include <memory>
#include <vector>

namespace megamol {
namespace core {

/** forward declaration */
class CoreInstance;

namespace utility {

/**
 * Abstract base class for service objects
 */
class ServiceManager {
public:
    /**
     * Callback to delete service objects
     *
     * @param The service to be deleted
     */
    typedef void (*ServiceDeletor)(AbstractService*&);

    /**
     * Ctor
     *
     * @param core The owning core instance
     */
    ServiceManager(CoreInstance& core);

    /** dtor */
    ~ServiceManager();

    /**
     * Installs a service object. The service object is initialized and potentially enabled
     *
     * @param service The service object to be installed
     *
     * @return 0 in case of an error. Larger zero is the service ID. If zero
     *         is returned, the caller is responsible for deleting the
     *         service object. Otherwise the core instance takes control of
     *         the memory.
     */
    unsigned int InstallServiceObject(AbstractService* service, ServiceDeletor deletor);

    /**
     * Answer the installed service object by it's ID.
     *
     * @param ID The id of the service object to be returned
     *
     * @return The installed service object with the provided ID or null if no such service exists.
     */
    AbstractService* GetInstalledService(unsigned int id);

private:
    /** The owning core instance */
    CoreInstance& core;

    /** All installed services */
    std::vector<std::shared_ptr<AbstractService>> services;
};

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif MEGAMOLCORE_SERVICEMANAGER_H_INCLUDED
