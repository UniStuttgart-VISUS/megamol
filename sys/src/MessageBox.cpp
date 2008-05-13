/*
 * MessageBox.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Sebastian Grottel. Alle Rechte vorbehalten.
 */

#include "vislib/MessageBox.h"
#include "vislib/IllegalStateException.h"
#include "vislib/MissingImplementationException.h"
#ifndef _WIN32
#include <cstdio>
#endif /* !_WIN32 */

#ifdef _WIN32
#ifdef _MSC_VER
#pragma push_macro("MessageBox")
#undef MessageBox
#else /* _MSC_VER */
#ifdef MessageBox
#error MessageBox Macro defined!
#endif /* MessageBox */
#endif /* _MSC_VER */
#endif /* _WIN32 */


/*
 * vislib::sys::MessageBox::Show
 */
vislib::sys::MessageBox::ReturnValue 
vislib::sys::MessageBox::Show(const vislib::StringA& msg,
        const vislib::StringA& title, MsgButtons btns, MsgIcon icon,
        DefButton defBtn) {
    return MessageBox(msg, title, btns, icon, defBtn).ShowDialog();
}


/*
 * vislib::sys::MessageBox::Show
 */
vislib::sys::MessageBox::ReturnValue 
vislib::sys::MessageBox::Show(const vislib::StringW& msg,
        const vislib::StringW& title, MsgButtons btns, MsgIcon icon,
        DefButton defBtn) {
    return MessageBox(msg, title, btns, icon, defBtn).ShowDialog();
}


/*
 * vislib::sys::MessageBox::ShowDialog
 */
vislib::sys::MessageBox::ReturnValue 
vislib::sys::MessageBox::ShowDialog(const vislib::StringA& msg,
        const vislib::StringA& title, MsgButtons btns, MsgIcon icon,
        DefButton defBtn) {
    return MessageBox(msg, title, btns, icon, defBtn).ShowDialog();
}


/*
 * vislib::sys::MessageBox::ShowDialog
 */
vislib::sys::MessageBox::ReturnValue 
vislib::sys::MessageBox::ShowDialog(const vislib::StringW& msg,
        const vislib::StringW& title, MsgButtons btns, MsgIcon icon,
        DefButton defBtn) {
    return MessageBox(msg, title, btns, icon, defBtn).ShowDialog();
}


/*
 * vislib::sys::MessageBox::MessageBox
 */
vislib::sys::MessageBox::MessageBox(const vislib::StringA &msg,
        const vislib::StringA &title, MsgButtons btns, MsgIcon icon,
        DefButton defBtn) : btns(btns), defBtn(defBtn), icon(icon), msg(msg),
        retval(RET_NONE), title(title) {
    // intentionally empty
}


/*
 * vislib::sys::MessageBox::MessageBox
 */
vislib::sys::MessageBox::MessageBox(const vislib::StringW &msg,
        const vislib::StringW &title, MsgButtons btns, MsgIcon icon,
        DefButton defBtn) : btns(btns), defBtn(defBtn), icon(icon), msg(msg),
        retval(RET_NONE), title(title) {
    // intentionally empty
}


/*
 * vislib::sys::MessageBox::~MessageBox
 */
vislib::sys::MessageBox::~MessageBox(void) {
    // intentionally empty
}


/*
 * vislib::sys::MessageBox::ShowDialog
 */
vislib::sys::MessageBox::ReturnValue
vislib::sys::MessageBox::ShowDialog(void) {
#ifdef _WIN32
    UINT type = 0;
    int rv;

    switch (this->btns) {
        case BTNS_OK: type |= MB_OK; break;
        case BTNS_OKCANCEL: type |= MB_OKCANCEL; break;
        case BTNS_RETRYCANCEL: type |= MB_RETRYCANCEL; break;
        case BTNS_YESNO: type |= MB_YESNO; break;
        case BTNS_YESNOCANCEL: type |= MB_YESNOCANCEL; break;
        case BTNS_ABORTRETRYIGNORE: type |= MB_ABORTRETRYIGNORE; break;
        case BTNS_CANCELRETRYCONTINUE: type |= MB_CANCELTRYCONTINUE; break;
        default : throw IllegalStateException(
                  "MessageBox Buttons had no valid value.",
                  __FILE__, __LINE__);
    }

    switch (this->defBtn) {
        case DEFBTN_1: type |= MB_DEFBUTTON1; break;
        case DEFBTN_2: type |= MB_DEFBUTTON2; break;
        case DEFBTN_3: type |= MB_DEFBUTTON3; break;
        default : /* nothing to do */ break;
    }

    switch (this->icon) {
        case ICON_ERROR: type |= MB_ICONERROR; break;
        case ICON_WARNING: type |= MB_ICONWARNING; break;
        case ICON_INFO: type |= MB_ICONINFORMATION; break;
        case ICON_QUESTION: type |= MB_ICONQUESTION; break;
        case ICON_NONE: // fall through
        default : /* nothing to do */ break;
    }

    rv = ::MessageBoxW(NULL, this->msg.PeekBuffer(), this->title.PeekBuffer(),
        type);

    switch (rv) {
        case IDOK: this->retval = RET_OK; break;
        case IDCANCEL: this->retval = RET_CANCEL; break;
        case IDYES: this->retval = RET_YES; break;
        case IDNO: this->retval = RET_NO; break;
        case IDRETRY: this->retval = RET_RETRY; break;
        case IDABORT: this->retval = RET_ABORT; break;
        case IDCONTINUE: this->retval = RET_CONTINUE; break;
        case IDIGNORE: this->retval = RET_IGNORE; break;
        case IDTRYAGAIN: // fall through
        default: this->retval = RET_NONE; break;
    }

#else /* _WIN32 */

    // try connect to the x server

    // TODO: Implement

    if (false) {
        // x server available. Use a GUI dialog

        // TODO: Implement

        throw MissingImplementationException("MessageBox::ShowDialog", __FILE__, __LINE__);

    } else {
        // x server unavailable. Use a console dialog (stdout/stdin)
        FILE *out = (this->icon == ICON_ERROR) ? stderr : stdout;

        // print title and message
        if (!this->title.IsEmpty()) {
            fprintf(out, "\n\t%s\n\n", vislib::StringA(this->title).PeekBuffer());
        }
        switch (this->icon) {
            case ICON_ERROR: fprintf(out, "Error: "); break;
            case ICON_WARNING: fprintf(out, "Warning: "); break;
            case ICON_INFO: // fall through
            case ICON_QUESTION: // fall through
            case ICON_NONE: // fall through
            default : /* nothing to do */ break;
        }
        fprintf(out, "%s\n\n", vislib::StringA(this->msg).PeekBuffer());

        // simple ok message boxes are handled separatly
        if (this->btns == BTNS_OK) {
            fprintf(out, "Hit \"Return\" to continue\n");
            char i;
            fscanf(stdin, "%c", &i);
            return this->retval = RET_OK;
        }

        // construct button captions
        const char *btn1 = NULL, *btn2 = NULL, *btn3 = NULL;
        switch (this->btns) {
            case BTNS_OKCANCEL: btn1 = "OK"; btn2 = "Cancel"; break;
            case BTNS_RETRYCANCEL: btn1 = "Retry"; btn2 = "Cancel"; break;
            case BTNS_YESNO: btn1 = "Yes"; btn2 = "No"; break;
            case BTNS_YESNOCANCEL: btn1 = "Yes"; btn2 = "No"; btn3 = "Cancel";
                break;
            case BTNS_ABORTRETRYIGNORE: btn1 = "Abort"; btn2 = "Retry";
                btn3 = "Ignore"; break;
            case BTNS_CANCELRETRYCONTINUE: btn1 = "Cancel"; btn2 = "Retry";
                btn3 = "Continue"; break;
            case BTNS_OK: // fall through
            default: ASSERT(false); break;
        }

        // message loop
        int answerBtn = -1;
        char answer[1024];
        bool b1, b2, b3;
        int pos;
        int l1, l2, l3, l;
        l1 = strlen(btn1);
        l2 = strlen(btn2);
        l3 = (btn3 == NULL) ? 0 : strlen(btn3);

        while (answerBtn < 0) {
            fprintf(out, (btn3 != NULL) 
                ? "Enter your answer (\"%s\"%s, \"%s\"%s, or \"%s\"%s): "
                : "Enter your answer (\"%s\"%s or \"%s\"%s): ",
                btn1, (this->defBtn == DEFBTN_1) ? "[Default]" : "",
                btn2, (this->defBtn == DEFBTN_2) ? "[Default]" : "",
                btn3, (this->defBtn == DEFBTN_3) ? "[Default]" : "");

            fscanf(stdin, "%s", answer);

            l = strlen(answer);

            b1 = b2 = true;
            b3 = (btn3 != NULL);
            for (pos = 0; pos < l; pos++) {
                if (pos >= l1) b1 = false;
                if (pos >= l2) b2 = false;
                if (pos >= l3) b3 = false;

                if (b1 && (tolower(btn1[pos]) != tolower(answer[pos]))) {
                    b1 = false;
                }
                if (b2 && (tolower(btn2[pos]) != tolower(answer[pos]))) {
                    b2 = false;
                }
                if (b3 && (tolower(btn3[pos]) != tolower(answer[pos]))) {
                    b3 = false;
                }
            }

            if (b1 && !b2 && !b3) { answerBtn = 1; }
            else if (!b1 && b2 && !b3) { answerBtn = 2; }
            else if (!b1 && !b2 && b3) { answerBtn = 3; }
            else if (!b1 && !b2 && !b3) {
                fprintf(out, "Invalid answer!\n");
            } else if (b1 && b2 && (b3 || (btn3 == NULL))) {
                switch (this->defBtn) {
                    case DEFBTN_1:
                        fprintf(out, "Answer \"%s\" chosen by default.\n",
                            btn1);
                        answerBtn = 1;
                        break;
                    case DEFBTN_2:
                        fprintf(out, "Answer \"%s\" chosen by default.\n",
                            btn2);
                        answerBtn = 2;
                        break;
                    case DEFBTN_3:
                        if (btn3 == NULL) {
                            fprintf(out, "Invalid answer!\n");
                        } else {
                            fprintf(out, "Answer \"%s\" chosen by default.\n",
                                btn3);
                            answerBtn = 3;
                        }
                        break;
                }
            } else {
                fprintf(out, "Ambiguous answer!\n");
            }

        }

        // set return value
        switch (this->btns) {
            case BTNS_OKCANCEL: 
                this->retval = (answerBtn == 1) ? RET_OK : RET_CANCEL;
                break;
            case BTNS_RETRYCANCEL: 
                this->retval = (answerBtn == 1) ? RET_RETRY : RET_CANCEL;
                break;
            case BTNS_YESNO: 
                this->retval = (answerBtn == 1) ? RET_YES : RET_NO;
                break;
            case BTNS_YESNOCANCEL: 
                switch (answerBtn) {
                    case 1: this->retval = RET_YES; break;
                    case 2: this->retval = RET_NO; break;
                    case 3: this->retval = RET_CANCEL; break;
                }
                break;
            case BTNS_ABORTRETRYIGNORE: btn1 = "Abort"; btn2 = "Retry";
                switch (answerBtn) {
                    case 1: this->retval = RET_ABORT; break;
                    case 2: this->retval = RET_RETRY; break;
                    case 3: this->retval = RET_IGNORE; break;
                }
                break;
            case BTNS_CANCELRETRYCONTINUE: btn1 = "Cancel"; btn2 = "Retry";
                switch (answerBtn) {
                    case 1: this->retval = RET_CANCEL; break;
                    case 2: this->retval = RET_RETRY; break;
                    case 3: this->retval = RET_CONTINUE; break;
                }
                break;
            case BTNS_OK: // fall through
            default: ASSERT(false); break;
        }

    }

#endif /* _WIN32 */

    return this->retval;
}


#ifdef _WIN32
#ifdef _MSC_VER
#pragma pop_macro("MessageBox")
#endif /* _MSC_VER */
#endif /* _WIN32 */
