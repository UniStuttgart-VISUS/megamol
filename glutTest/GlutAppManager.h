/*
 * AbstractGlutApp.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_GLUTAPPMANAGER_H_INCLUDED
#define VISLIBTEST_GLUTAPPMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGlutApp.h"
#include "vislib/String.h"
#include "vislib/Array.h"

/**
 * Management class for glut applications.
 * Singelton class implementation.
 */
class GlutAppManager {
public:

    /**
     * Nested base class for factory objects.
     */
    class AbstractFactory {
    public:

        /**
         * ctor.
         *
         * @param name An ANSI string holding the name of the test application.
         *             This will also be used as menu item text.
         */
        AbstractFactory(const char *name);

        /** dtor. */
        virtual ~AbstractFactory(void) { }

        /**
         * Factory method.
         *
         * @return Pointer to a new application object created with "new".
         */
        virtual AbstractGlutApp * CreateApplication(void) = 0;

        /**
         * Checks whether a given application has been created by this factory.
         *
         * @param app The application object to test.
         *
         * @return true if the application has been created by this factory,
         *         false otherwise.
         */
        virtual bool HasCreated(AbstractGlutApp *app) = 0;

        /**
         * Answer the name of the factory.
         *
         * @return The name of the factory as zero terminated ANSI string.
         */
        inline const char* GetName(void) const {
            return this->name;
        }

    private:
        /** the name */
        vislib::StringA name;

    };

    /**
     * Nested template class for factory objects.
     * Instanciate with T derived class from AbstractGlutApplication.
     */
    template<class T> class Factory: public AbstractFactory {
    public:

        /**
         * ctor.
         *
         * @param name An ANSI string holding the name of the test application.
         *             This will also be used as menu item text.
         */
        Factory(const char *name);

        /** dtor. */
        virtual ~Factory(void) { }

        /**
         * Factory methode.
         *
         * @return Pointer to a new application object created with "new".
         */
        virtual AbstractGlutApp * CreateApplication(void);

        /**
         * Checks whether a given application has been created by this factory.
         *
         * @param app The application object to test.
         *
         * @return true if the application has been created by this factory,
         *         false otherwise.
         */
        virtual bool HasCreated(AbstractGlutApp *app);

    };

    /**
     * Singelton function.
     *
     * @return Pointer to the only instance of this class.
     */
    static GlutAppManager * GetInstance(void);

    /**
     * Shortcut to the current Application.
     *
     * @return Pointer to the current selected Test application.
     */
    static inline AbstractGlutApp * CurrentApp(void) {
        return GetInstance()->app;
    }

    /**
     * The glut menu callback function.
     *
     * @param menuID The id of the selected menu entry
     */
    static void OnMenuItemClicked(int menuID);

    /**
     * Installs a factory object into the manager. The factory object has to be
     * created with "new". The manager takes ownership of the object and will
     * release it at programm termination.
     *
     * @param factory The new factory object to be installed.
     */
    static void InstallFactory(AbstractFactory *factory);

    /**
     * Installs a factory object into the manager. The factory object has to be
     * created with "new". The manager takes ownership of the object and will
     * release it at programm termination.
     *
     * @param name An ANSI string holding the name of the test application.
     *             This will also be used as menu item text.
     */
    template <class Tp> static inline void InstallFactory(const char *name);

    /**
     * Initializes the glut window by adding a menu. All calls to 
     * "InstallFactory" must be performed before calling this methode.
     **/
    void InitGlutWindow(void);

    /**
     * Terminates this glut application.
     *
     * @param exitcode The return value to be returned by this application if possible.
     */
    static void ExitApplication(int exitcode);

    /**
     * Sets the size of the glut viewport.
     *
     * @param w The new width.
     * @param h The new height.
     */
    void SetSize(int w, int h);

    /** Renders an welcome text into the current open gl context. */
    void glRenderEmptyScreen(void);

private:

    /** private ctor */
    GlutAppManager(void);

    /** private dtor */
    ~GlutAppManager(void);

    /** pointer to the Current Application. */
    AbstractGlutApp *app;

    /** Array of all installed factories */
    vislib::Array<AbstractFactory *> factories;

    /** Identifier of the window menu */
    int windowMenu;

    /** Identifier of the test selection menu */
    int appMenu;

    /** The width of the glut viewport */
    int width;

    /** The height of the glut viewport */
    int height;

};


/*
 * GlutAppManager::Factory<T>::Factory
 */
template <class T> GlutAppManager::Factory<T>::Factory(const char *name) : GlutAppManager::AbstractFactory(name) {
}


/*
 * GlutAppManager::Factory<T>::CreateApplication
 */
template <class T> AbstractGlutApp * GlutAppManager::Factory<T>::CreateApplication(void) {
    return new T;
}


/*
 * GlutAppManager::Factory<T>::HasCreated
 */
template <class T> bool GlutAppManager::Factory<T>::HasCreated(AbstractGlutApp *app) {
    return dynamic_cast<T*>(app) != NULL;
}


/*
 * GlutAppManager::InstallFactory
 */
template <class Tp> void GlutAppManager::InstallFactory(const char *name) {
    GlutAppManager::InstallFactory(new GlutAppManager::Factory<Tp>(name));
}

#endif /* VISLIBTEST_GLUTAPPMANAGER_H_INCLUDED */
