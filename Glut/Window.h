/*
 * Window.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_WINDOW_H_INCLUDED
#define MEGAMOLVIEWER_WINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "MegaMolViewer.h"
#include "ApiHandle.h"
#include "CallbackSlot.h"
#include "Viewer.h"
#include "vislib/String.h"


namespace megamol {
namespace viewer {

    /**
     * class of OpenGL windows.
     */
    class Window : public ApiHandle {
    public:

        /**
         * Ctor.
         * This also creates a new OpenGL context for the window and sets this
         * context active.
         *
         * @param owner The viewer owning this window.
         */
        Window(megamol::viewer::Viewer& owner);

        /** Dtor. */
        virtual ~Window(void);

        /** Closes this window */
        void Close(void);

        /**
         * Gets the callback slot requested.
         *
         * @param slot The slot to be returned.
         *
         * @return The requested callback slot or NULL in case of an error.
         */
        inline CallbackSlot* Callback(mmvWindowCallbacks slot) {
            switch (slot) {
                case MMV_WINCB_RENDER: return &this->renderCallback;
                case MMV_WINCB_RESIZE: return &this->resizeCallback;
                case MMV_WINCB_KEY: return &this->keyCallback;
                case MMV_WINCB_MOUSEBUTTON: return &this->mouseButtonCallback;
                case MMV_WINCB_MOUSEMOVE: return &this->mouseMoveCallback;
                case MMV_WINCB_CLOSE: return &this->closeCallback;
                case MMV_WINCB_COMMAND: return &this->commandCallback;
                case MMV_WINCB_APPEXIT: return &this->owner.ApplicationTerminateCallbackSlot();
                case MMV_WINCB_UPDATEFREEZE: return &this->updateFreeze;
                default: return NULL;
            }
            return NULL;
        }

        /**
         * Gets the render callback slot.
         *
         * @return The render callback slot.
         */
        inline CallbackSlot& RenderCallback(void) {
            return this->renderCallback;
        }

        /**
         * Gets the resize callback slot.
         *
         * @return The resize callback slot.
         */
        inline CallbackSlot ResizeCallback(void) {
            return this->resizeCallback;
        }

        /**
         * Gets the key callback slot.
         *
         * @return The key callback slot.
         */
        inline CallbackSlot KeyCallback(void) {
            return this->keyCallback;
        }

        /**
         * Gets the mouse button callback slot.
         *
         * @return The mouse button callback slot.
         */
        inline CallbackSlot MouseButtonCallback(void) {
            return this->mouseButtonCallback;
        }

        /**
         * Gets the mouse move callback slot.
         *
         * @return The mouse move callback slot.
         */
        inline CallbackSlot MouseMoveCallback(void) {
            return this->mouseMoveCallback;
        }

        /**
         * Gets the close callback slot.
         *
         * @return The close callback slot.
         */
        inline CallbackSlot CloseCallback(void) {
            return this->closeCallback;
        }

        /**
         * Installs a context menu.
         */
        void InstallContextMenu(void);

        /**
         * Resizes the window.
         *
         * @param width The new width
         * @param height The new height
         */
        void ResizeWindow(unsigned int width, unsigned int height);

        /**
         * Moves the window to the given coordinates.
         *
         * @param x The new x coordinate
         * @param y The new y coordinate
         */
        void MoveWindowTo(int x, int y);

        /**
         * Adds a command menu item to the context menu.
         *
         * @param caption The caption for the command menu item.
         * @param value The value of the command.
         */
        void AddCommand(const char *caption, int value);

        /**
         * Sets the title of the window.
         *
         * @param title The new title.
         */
        void SetTitle(const char *title);

        /**
         * Sets the title of the window.
         *
         * @param title The new title.
         */
        void SetTitle(const wchar_t *title);

        /**
         * Sets this window into fullscreen mode.
         */
        void SetFullscreen(void);

        /**
         * Activates or deactivates the window decorations.
         *
         * @param dec 'true' to activate, 'false' to deactivate the window
         *            decorations
         */
        void ShowDecorations(bool dec);

        /**
         * Shows or hides the mouse cursor in the window.
         *
         * @param visible Visibility flag of the mouse cursor
         */
        void SetCursorVisibility(bool visible);

        /**
         * Flags the window to stay on top or not.
         *
         * @param stay 'true' window stay on top, 'false' window normally
         *             ordered in z-depth.
         */
        void StayOnTop(bool stay);

        /**
         * De-/activates the presentation mode for this window
         *
         * @param presentation The new value for the presentation mode flag
         */
        void SetPresentationMode(bool presentation);

        /**
         * Answer the presentation mode value of the window.
         *  0  window is not in presentation mode
         *  1  window is in presentation mode
         *  2  window is in presentation mode and needs a buffer swap
         *
         * @return The presentation mode value of the window
         */
        inline unsigned int GetPresentationMode(void) const {
            return this->presentationMode;
        }

        /**
         * Performs a presentation mode buffer swap
         */
        void PresentationModeSwap(void);

        /**
         * Performs a presentation mode buffer swap
         */
        void PresentationModeUpdate(void);

        /**
         * Performs a presentation mode buffer swap
         */
        void PresentationModeRefresh(void);

    private:

        /**
         * Access to the 'Window' object from within the glut callbacks.
         *
         * @return The 'Window' object.
         */
        static Window* thisWindow(void);

        /**
         * The glut display callback.
         */
        static void glutDisplayCallback(void);

        /**
         * The glut reshape callback.
         *
         * @param w The new width;
         * @param h The new height;
         */
        static void glutReshapeCallback(int w, int h);

        /**
         * The glut keyboard callback.
         *
         * @param key The ASCII char of the pressed key.
         * @param x The x coordinate of the mouse cursor.
         * @param y The y coordinate of the mouse cursor.
         */
        static void glutKeyboardCallback(unsigned char key, int x, int y);

        /**
         * The glut special key callback.
         *
         * @param key The key code of the key pressed.
         * @param x The x coordinate of the mouse cursor.
         * @param y The y coordinate of the mouse cursor.
         */
        static void glutSpecialCallback(int key, int x, int y);

        /**
         * The glut mouse callback.
         *
         * @param button The mouse button pressed or released.
         * @param state The new state of the mouse button.
         * @param x The x coordinate of the mouse cursor.
         * @param y The y coordinate of the mouse cursor.
         */
        static void glutMouseCallback(int button, int state, int x, int y);

        /**
         * The glut mouse motion callback.
         *
         * @param x The x coordinate of the mouse cursor.
         * @param y The y coordinate of the mouse cursor.
         */
        static void glutMotionCallback(int x, int y);

        /**
         * The glut close callback.
         */
        static void glutCloseCallback(void);

        /**
         * The glut main menu callback.
         *
         * @param item The id of the menu item clicked.
         */
        static void glutMainMenuCallback(int item);

        /**
         * The glut size menu callback.
         *
         * @param item The id of the menu item clicked.
         */
        static void glutSizeMenuCallback(int item);

        /**
         * The glut main menu callback.
         *
         * @param item The id of the menu item clicked.
         */
        static void glutCommandMenuCallback(int item);

        /**
         * Updates the glut size menu entry for the fullscreen mode.
         */
        void updateFullscreenMenuItem(void) const;

        /**
         * Toggles window mode and fullscreen.
         */
        void toogleFullscreen(void);

        /** the glut id of the window */
        int glutID;

        /** The viewer owning this window. */
        megamol::viewer::Viewer& owner;

        /** The x coordinate of the upper left corner. */
        int left;

        /** The y coordinate of the upper left corner */
        int top;

        /** The width of the window. */
        unsigned int width;

        /** The height of the window. */
        unsigned int height;

        /** Whether or not this window is in fullscreen mode */
        bool isFullscreen;

        /** The render callback slot */
        CallbackSlot renderCallback;

        /** The resize callback slot */
        CallbackSlot resizeCallback;

        /** The key callback slot */
        CallbackSlot keyCallback;

        /** The mouse button callback slot */
        CallbackSlot mouseButtonCallback;

        /** The mouse move callback slot */
        CallbackSlot mouseMoveCallback;

        /** The close callback slot */
        CallbackSlot closeCallback;

        /** The command callback slot */
        CallbackSlot commandCallback;

        /** The update freeze slot */
        CallbackSlot updateFreeze;

        /** The glut main menu id */
        int glutMainMenu;

        /** The glut size menu id */
        int glutSizeMenu;

        /** The glut command menu id */
        int glutCommandMenu;

        /** The glut modifier flags */
        int modifiers;

        /** The presentation mode flag */
        int presentationMode;

    };

} /* end namespace viewer */
} /* end namespace megamol */

#endif /* MEGAMOLVIEWER_WINDOW_H_INCLUDED */

