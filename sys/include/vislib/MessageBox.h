/*
 * MessageBox.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Sebastian Grottel. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MESSAGEBOX_H_INCLUDED
#define VISLIB_MESSAGEBOX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/String.h"


namespace vislib {
namespace sys {

#ifdef _WIN32
    // Crowbar to work around windows naming conflict
#ifdef _MSC_VER
#pragma push_macro("MessageBox")
#undef MessageBox
#else /* _MSC_VER */
#ifdef MessageBox
#error MessageBox Macro defined!
#endif /* MessageBox */
#endif /* _MSC_VER */
#endif /* _WIN32 */

    /**
     * Class of modal message box dialog windows. If graphical dialogs are not
     * supported (e.g. on Linux without x-server) stdout/stderr and stdin are
     * used.
     */
    class MessageBox {
    public:

        /** possible buttons */
        enum MsgButtons {
            BTNS_OK,
            BTNS_OKCANCEL,
            BTNS_RETRYCANCEL,
            BTNS_YESNO,
            BTNS_YESNOCANCEL,
            BTNS_ABORTRETRYIGNORE,
            BTNS_CANCELRETRYCONTINUE
        };

        /** possible default button flags */
        enum DefButton {
            DEFBTN_1,
            DEFBTN_2,
            DEFBTN_3
        };

        /** possible icons */
        enum MsgIcon {
            ICON_NONE,    // no icon
            ICON_ERROR,   // red cross
            ICON_WARNING, // yellow exclamation mark
            ICON_INFO,    // blue 'i'
            ICON_QUESTION // blue '?'
        };

        /** possible return values */
        enum ReturnValue {
            RET_NONE, // used to indicate that the dialog was not yet closed.
            RET_OK,
            RET_CANCEL,
            RET_YES,
            RET_NO,
            RET_RETRY,
            RET_ABORT,
            RET_CONTINUE,
            RET_IGNORE
        };

        /**
         * Shows the modal message box dialog.
         *
         * @param msg The message to be shown.
         * @param title The title of the message box.
         * @param btns The buttons to be shown.
         * @param icon The icon to be shown.
         * @param defBtn The default button.
         *
         * @return The return value of the dialog.
         */
        static ReturnValue Show(const vislib::StringA& msg,
            const vislib::StringA& title, MsgButtons btns = BTNS_OK,
            MsgIcon icon = ICON_NONE, DefButton defBtn = DEFBTN_1);

        /**
         * Shows the modal message box dialog.
         *
         * @param msg The message to be shown.
         * @param title The title of the message box.
         * @param btns The buttons to be shown.
         * @param icon The icon to be shown.
         * @param defBtn The default button.
         *
         * @return The return value of the dialog.
         */
        static ReturnValue Show(const vislib::StringW& msg,
            const vislib::StringW& title, MsgButtons btns = BTNS_OK,
            MsgIcon icon = ICON_NONE, DefButton defBtn = DEFBTN_1);

        /**
         * Shows the modal message box dialog.
         *
         * @param msg The message to be shown.
         * @param title The title of the message box.
         * @param btns The buttons to be shown.
         * @param icon The icon to be shown.
         * @param defBtn The default button.
         *
         * @return The return value of the dialog.
         */
        static ReturnValue ShowDialog(const vislib::StringA& msg,
            const vislib::StringA& title, MsgButtons btns = BTNS_OK,
            MsgIcon icon = ICON_NONE, DefButton defBtn = DEFBTN_1);

        /**
         * Shows the modal message box dialog.
         *
         * @param msg The message to be shown.
         * @param title The title of the message box.
         * @param btns The buttons to be shown.
         * @param icon The icon to be shown.
         * @param defBtn The default button.
         *
         * @return The return value of the dialog.
         */
        static ReturnValue ShowDialog(const vislib::StringW& msg,
            const vislib::StringW& title, MsgButtons btns = BTNS_OK,
            MsgIcon icon = ICON_NONE, DefButton defBtn = DEFBTN_1);

        /**
         * Ctor.
         *
         * @param msg The message to be shown.
         * @param title The title of the message box.
         * @param btns The buttons to be shown.
         * @param icon The icon to be shown.
         * @param defBtn The default button.
         */
        MessageBox(const vislib::StringA& msg, const vislib::StringA& title,
            MsgButtons btns = BTNS_OK, MsgIcon icon = ICON_NONE,
            DefButton defBtn = DEFBTN_1);

        /**
         * Ctor.
         *
         * @param msg The message to be shown.
         * @param title The title of the message box.
         * @param btns The buttons to be shown.
         * @param icon The icon to be shown.
         * @param defBtn The default button.
         */
        MessageBox(const vislib::StringW& msg, const vislib::StringW& title,
            MsgButtons btns = BTNS_OK, MsgIcon icon = ICON_NONE,
            DefButton defBtn = DEFBTN_1);

        /** Dtor. */
        ~MessageBox(void);

        /**
         * Answer the message buttons to be used.
         *
         * @return The message buttons to be used.
         */
        inline MsgButtons Buttons(void) const {
            return this->btns;
        }

        /**
         * Answer the default button.
         *
         * @return The default button.
         */
        inline DefButton DefaultButton(void) const {
            return this->defBtn;
        }

        /**
         * Answer the return value of the message box.
         *
         * @return The return value of the message box.
         */
        inline ReturnValue GetReturnValue(void) const {
            return this->retval;
        }

        /**
         * Answer the icon.
         *
         * @return The icon.
         */
        inline MsgIcon Icon(void) const {
            return this->icon;
        }

        /**
         * Answer the message text of the message box.
         *
         * @return The message text of the message box.
         */
        inline const vislib::StringW& Message(void) const {
            return this->msg;
        }

        /**
         * Sets the buttons to be used.
         *
         * @param btns The buttons to be used.
         */
        inline void SetButtons(MsgButtons btns) {
            this->btns = btns;
        }

        /**
         * Sets the default button.
         *
         * @param defBtn The default button.
         */
        inline void SetDefaultButton(DefButton defBtn) {
            this->defBtn = defBtn;
        }

        /**
         * Sets the icon to be used.
         *
         * @return icon The icon to be used.
         */
        inline void SetIcon(MsgIcon icon) {
            this->icon = icon;
        }

        /**
         * Sets the message text of the message box.
         *
         * @param msg The message text of the message box.
         */
        inline void SetMessage(const vislib::StringA& msg) {
            this->msg = msg;
        }

        /**
         * Sets the message text of the message box.
         *
         * @param msg The message text of the message box.
         */
        inline void SetMessage(const vislib::StringW& msg) {
            this->msg = msg;
        }

        /**
         * Sets the title of the message box.
         *
         * @param msg The title of the message box.
         */
        inline void SetTitle(const vislib::StringA& title) {
            this->title = title;
        }

        /**
         * Sets the title of the message box.
         *
         * @param msg The title of the message box.
         */
        inline void SetTitle(const vislib::StringW& title) {
            this->title = title;
        }

        /**
         * Shows the modal message box dialog. This method will not return
         * until the user clicks on one of the shown buttons.
         *
         * @return The return value of the dialog.
         */
        inline ReturnValue Show(void) {
            return this->ShowDialog();
        }

        /**
         * Shows the modal message box dialog. This method will not return
         * until the user clicks on one of the shown buttons.
         *
         * @return The return value of the dialog.
         */
        ReturnValue ShowDialog(void);

        /**
         * Answer the title of the message box.
         *
         * @return The title of the message box.
         */
        inline const vislib::StringW& Title(void) const {
            return this->title;
        }

    private:

        /** The message buttons to be shown. */
        MsgButtons btns;

        /** The default button */
        DefButton defBtn;

        /** The icon to be used */
        MsgIcon icon;

        /** The message of the message box */
        vislib::StringW msg;

        /** The return value of the message box. */
        ReturnValue retval;

        /** The title of the message box. */
        vislib::StringW title;

    };

#ifdef _WIN32

    /** type definition to work around windows macro naming conflict */
    typedef MessageBox MessageBoxW;

    /** type definition to work around windows macro naming conflict */
    typedef MessageBox MessageBoxA;

#ifdef _MSC_VER
#pragma pop_macro("MessageBox")
#endif /* _MSC_VER */
#endif /* _WIN32 */

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MESSAGEBOX_H_INCLUDED */

