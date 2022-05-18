/*
 * AbstractService.h
 *
 * Copyright (C) 2016 by MegaMol Team (S. Grottel)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_ABSTRACTSERVICE_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSERVICE_H_INCLUDED
#pragma once

namespace megamol {
namespace core {

/** forward declaration */
class CoreInstance;

/**
 * Abstract base class for service objects
 */
class AbstractService {
public:
    /**
     * dtor
     *
     * Deinitialize is ensured to be called before.
     */
    virtual ~AbstractService();

    /**
     * Answer the name of the service.
     *
     * @return The name of the service
     */
    virtual const char* Name() const = 0;

    /**
     * Initializes the service.
     * The default implementation does nothing.
     *
     * @param autoEnable If true is returned the service is enabled shortly after
     *
     * @return True on success. In case of error, messages should be logged.
     */
    virtual bool Initalize(bool& autoEnable);

    /**
     * Deinitializes the service at program exit
     * The default implementation disables the service, if it is still enabled.
     *
     * @return True on success. In case of error, messages should be logged.
     */
    virtual bool Deinitialize();

    /**
     * Enables this service
     *
     * @return True on success. In case of error, messages should be logged.
     */
    inline bool Enable() {
        if (!enabled) {
            if (enableImpl()) {
                enabled = true;
            }
        }
        return enabled;
    }

    /**
     * Disables this service
     *
     * @return True on success. In case of error, messages should be logged.
     */
    inline bool Disable() {
        if (enabled) {
            if (disableImpl()) {
                enabled = false;
            }
        }
        return !enabled;
    }

    /**
     * Answer whether this service enabled.
     *
     * @return true if this service is enabled.
     */
    inline bool IsEnabled() const {
        return enabled;
    }

protected:
    /**
     * Ctor
     *
     * @param core The owning core instance
     */
    AbstractService(CoreInstance& core);

    /**
     * Enables this service
     *
     * @return True on success. In case of error, messages should be logged.
     */
    virtual bool enableImpl() = 0;

    /**
     * Disables this service
     *
     * @return True on success. In case of error, messages should be logged.
     */
    virtual bool disableImpl() = 0;

    /**
     * Access the owning core instance
     *
     * @return The owning core instance
     */
    inline CoreInstance& GetCoreInstance(void) const {
        return core;
    }

private:
    /** The owning core instance */
    CoreInstance& core;

    /** The enabled state of the service */
    bool enabled;
};

} /* end namespace core */
} /* end namespace megamol */

#endif MEGAMOLCORE_ABSTRACTSERVICE_H_INCLUDED
