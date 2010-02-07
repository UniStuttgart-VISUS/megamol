/*
 * GUILayer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#ifdef WITH_TWEAKBAR
#ifndef _MEGAMOL_CONSOLE_GUILAYER_H_INCLUDED
#define _MEGAMOL_CONSOLE_GUILAYER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#define TW_STATIC
#define TW_NO_LIB_PRAGMA
#include "AntTweakBar.h"
#include "CoreHandle.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"


namespace megamol {
namespace console {

    /**
     * Manager class for GUI layer
     */
    class GUILayer {
    public:

        /**
         * User class implementing a GUI layer factory with reference counting
         */
        class GUIClient {
        private:

            /* Forword declaration */
            class Parameter;

        public:

            /**
             * The factory method
             *
             * @param bar The bar handle
             * @param hParam The parameter core handle
             * @param name The name of the parameter as zero-terminated ANSI string
             * @param desc The parameter description
             * @param len The length of the description in bytes
             *
             * @return The parameter object representing the parameter
             */
            static Parameter *Factory(TwBar *bar,
                vislib::SmartPtr<megamol::console::CoreHandle> hParam,
                    const char *name, unsigned char *desc, unsigned int len);

            /** Ctor */
            GUIClient(void);

            /** Dtor */
            ~GUIClient(void);

            /**
             * Grants access to the gui layer
             *
             * @return The gui layer object
             */
            GUILayer& Layer(void);

            /**
             * Activates this client
             */
            void Activate(void);

            /**
             * Deactivates this client
             */
            void Deactivate(void);

            /**
             * Tells the gui client the window size
             *
             * @param w The width of the window
             * @param h The height of the window
             */
            void SetWindowSize(unsigned int w, unsigned int h);

            /**
             * Adds a core parameter to the gui client
             *
             * @param hParam The parameter core handle
             * @param name The name of the parameter as zero-terminated ANSI string
             * @param desc The parameter description
             * @param len The length of the description in bytes
             */
            void AddParameter(vislib::SmartPtr<megamol::console::CoreHandle> hParam,
                const char *name, unsigned char *desc, unsigned int len);

            /** Draws the GUI */
            void Draw(void);

            /**
             * Informs the GUI that the mouse moved
             *
             * @param x The new mouse position
             * @param y The new mouse position
             *
             * @return True if the event was consumed by the gui
             */
            bool MouseMove(int x, int y);

            /**
             * Informs the GUI that a mouse button state changed
             *
             * @param btn The mouse button
             * @param down The new state flag
             *
             * @return True if the event was consumed by the gui
             */
            bool MouseButton(int btn, bool down);

            /**
             * Informs the GUI that a key has been pressed
             *
             * @param keycode The vislib key code
             * @param shift The shift modifier flag
             * @param alt The alt modifier flag
             * @param ctrl The control modifier flag
             *
             * @return True if the event was consumed by the gui
             */
            bool KeyPressed(unsigned short keycode, bool shift, bool alt, bool ctrl);

        private:

            /**
             * abstract base class for parameter objects
             */
            class Parameter {
            public:

                /**
                 * Dtor
                 */
                virtual ~Parameter(void) {
                    // intentionally empty
                }

                /**
                 * Answer the handle of the parameter
                 *
                 * @return The handle of the parameter
                 */
                inline vislib::SmartPtr<megamol::console::CoreHandle> CoreHandle(void) const {
                    return this->hndl;
                }

                /**
                 * Answer the handle of the parameter
                 *
                 * @return The handle of the parameter
                 */
                inline const megamol::console::CoreHandle& Handle(void) const {
                    return *this->hndl;
                }

                /**
                 * Answer the name of the parameter
                 *
                 * @return The name of the parameter
                 */
                inline const vislib::StringA& Name(void) const {
                    return this->name;
                }

            protected:

                /**
                 * Ctor
                 *
                 * @param bar The bar handle
                 * @param hParam The parameter core handle
                 * @param name The name of the parameter as zero-terminated ANSI string
                 * @param desc The parameter description
                 * @param len The length of the description in bytes
                 */
                Parameter(TwBar *bar, vislib::SmartPtr<megamol::console::CoreHandle> hParam,
                        const char *name)
                        : bar(bar), hndl(hParam), name(name) {
                    // intentionally empty
                }

                /**
                 * Answer the name for the object
                 *
                 * @return The name for the object
                 */
                vislib::StringA objName(void) const {
                    vislib::StringA n;
                    UINT64 id = reinterpret_cast<UINT64>(this);
                    unsigned char idc[8];
                    ::memcpy(idc, &id, 8);
                    n.Format("%.2x%.2x%.2x%.2x%.2x%.2x%.2x%.2x",
                        idc[0], idc[1], idc[2], idc[3],
                        idc[4], idc[5], idc[6], idc[7]);
                    return n;
                }

                /**
                 * Answer the gui bar
                 *
                 * @return the gui bar
                 */
                inline TwBar *Bar(void) const {
                    return this->bar;
                }

                /**
                 * Answer the name part of 'name'
                 *
                 * @param name The name of a core parameter
                 *
                 * @return The name part of 'name'
                 */
                static vislib::StringA paramName(const char *name);

                /**
                 * Answer the group part of 'name'
                 *
                 * @param name The name of a core parameter
                 *
                 * @return The group part of 'name'
                 */
                static vislib::StringA paramGroup(const char *name);

            private:

                /** The gui bar */
                TwBar *bar;

                /** The parameter core handle */
                vislib::SmartPtr<megamol::console::CoreHandle> hndl;

                /** The name of the parameter */
                vislib::StringA name;

            };

            /**
             * A parameter to store all initialization values in case the gui
             * is not yet ready
             */
            class PlaceboParameter : public Parameter {
            public:

                /**
                 * Ctor
                 *
                 * @param bar The bar handle
                 * @param hParam The parameter core handle
                 * @param name The name of the parameter as zero-terminated ANSI string
                 * @param desc The parameter description
                 * @param len The length of the description in bytes
                 */
                PlaceboParameter(vislib::SmartPtr<megamol::console::CoreHandle> hParam,
                        const char *name, unsigned char *desc, unsigned int len)
                        : Parameter(NULL, hParam, name) {
                    this->desc = new unsigned char[len];
                    ::memcpy(this->desc, desc, len);
                }

                /**
                 * Dtor.
                 */
                virtual ~PlaceboParameter(void) {
                    ARY_SAFE_DELETE(this->desc);
                    this->len = 0;
                }

                /**
                 * Answer a pointer to the stored description
                 *
                 * @return A pointer to the stored description
                 */
                inline unsigned char *Description(void) const {
                    return this->desc;
                }

                /**
                 * Answer the length of the description in bytes
                 *
                 * @return The length of the description in bytes
                 */
                inline unsigned int DescriptionLength(void) const {
                    return this->len;
                }

            private:

                /** The parameter description */
                unsigned char *desc;

                /** The length of the description in bytes */
                unsigned int len;

            };

            /**
             * Abstract base class for value parameters
             */
            class ValueParameter : public Parameter {
            public:

                /**
                 * The parameter set callback
                 *
                 * @param value The value
                 * @param clientData The client data
                 */
                static void TW_CALL Set(const void *value, void *clientData);

                /**
                 * The parameter set callback
                 *
                 * @param value The value
                 * @param clientData The client data
                 */
                static void TW_CALL Get(void *value, void *clientData);

                /**
                 * Ctor
                 *
                 * @param bar The bar handle
                 * @param hParam The parameter core handle
                 * @param name The name of the parameter as zero-terminated ANSI string
                 * @param desc The parameter description
                 * @param len The length of the description in bytes
                 * @param def The definitions used for the bar
                 */
                ValueParameter(TwBar *bar,
                    vislib::SmartPtr<megamol::console::CoreHandle> hParam,
                    TwType type, const char *name, unsigned char *desc,
                    unsigned int len, const char *def);

                /**
                 * Dtor.
                 */
                virtual ~ValueParameter(void);

            protected:

                /**
                 * The parameter set callback
                 *
                 * @param value The value
                 * @param clientData The client data
                 */
                virtual void Set(const void *value) = 0;

                /**
                 * The parameter set callback
                 *
                 * @param value The value
                 * @param clientData The client data
                 */
                virtual void Get(void *value) = 0;

            };

            /**
             * String parameter
             */
            class StringParameter : public ValueParameter {
            public:

                /**
                 * Ctor
                 *
                 * @param bar The bar handle
                 * @param hParam The parameter core handle
                 * @param name The name of the parameter as zero-terminated ANSI string
                 * @param desc The parameter description
                 * @param len The length of the description in bytes
                 */
                StringParameter(TwBar *bar, vislib::SmartPtr<megamol::console::CoreHandle> hParam,
                        const char *name, unsigned char *desc, unsigned int len);

                /**
                 * Dtor.
                 */
                virtual ~StringParameter(void);

            protected:

                /**
                 * The parameter set callback
                 *
                 * @param value The value
                 * @param clientData The client data
                 */
                virtual void Set(const void *value);

                /**
                 * The parameter set callback
                 *
                 * @param value The value
                 * @param clientData The client data
                 */
                virtual void Get(void *value);

            };

            /** the only instance of the gui layer */
            static GUILayer* layer;

            /** The reference counter */
            static SIZE_T cntr;

            /** The currently active client */
            static GUIClient* activeClient;

            /**
             * Answer the name for the bar
             *
             * @return The name for the bar
             */
            vislib::StringA name(void) const {
                vislib::StringA n;
                UINT64 id = reinterpret_cast<UINT64>(this);
                unsigned char idc[8];
                ::memcpy(idc, &id, 8);
                n.Format("%.2x%.2x%.2x%.2x%.2x%.2x%.2x%.2x",
                    idc[0], idc[1], idc[2], idc[3],
                    idc[4], idc[5], idc[6], idc[7]);
                return n;
            }

            /**
             * Answer the bar of the gui client
             *
             * @return The bar of the gui client
             */
            TwBar *myBar(void);

            /** The width of the window */
            int width;

            /** The height of the window */
            int height;

            /** The parameter bar of this client */
            TwBar *_myBar;

            /** The list of parameters */
            vislib::SingleLinkedList<Parameter *> params;

        };

        /** Friend factory may access the object */
        friend class GUIClient;

    private:

        /** Ctor */
        GUILayer(void);

        /** Dtor */
        ~GUILayer(void);

        /** Draws the GUI */
        void Draw(void);

        /**
         * Informs the GUI that the mouse moved
         *
         * @param x The new mouse position
         * @param y The new mouse position
         *
         * @return True if the event was consumed by the gui
         */
        bool MouseMove(int x, int y);

        /**
         * Informs the GUI that a mouse button state changed
         *
         * @param btn The mouse button
         * @param down The new state flag
         *
         * @return True if the event was consumed by the gui
         */
        bool MouseButton(int btn, bool down);

        /**
         * Informs the GUI that a key has been pressed
         *
         * @param keycode The vislib key code
         * @param shift The shift modifier flag
         * @param alt The alt modifier flag
         * @param ctrl The control modifier flag
         *
         * @return True if the event was consumed by the gui
         */
        bool KeyPressed(unsigned short keycode, bool shift, bool alt, bool ctrl);

        /** The active flag */
        bool active;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* _MEGAMOL_CONSOLE_GUILAYER_H_INCLUDED */
#endif /* WITH_TWEAKBAR */
