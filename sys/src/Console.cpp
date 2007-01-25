/*
 * Console.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/Console.h"

#include <cstdarg>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>

#include "DynamicFunctionPointer.h"

#else /* _WIN32 */
#include <stdlib.h>

#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <curses.h>
#include <term.h>

#include "vislib/StringConverter.h"

#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/Thread.h"


/*
 * vislib::sys::Console::usecolors
 */
bool vislib::sys::Console::useColors = vislib::sys::Console::ColorsAvailable();


/* 
 * vislib::sys::Console::defaultFgcolor
 */
vislib::sys::Console::ColorType vislib::sys::Console::defaultFgcolor = vislib::sys::Console::GetForegroundColor();


/* 
 * vislib::sys::Console::defaultBgcolor
 */
vislib::sys::Console::ColorType vislib::sys::Console::defaultBgcolor = vislib::sys::Console::GetBackgroundColor();


#ifndef _WIN32
/*
 * Helper class for initializing linux term
 */
class vislib::sys::Console::Curser {
public:

#define READER_DATA_BUFFER_SIZE 1024

    /** helper data struct for the stdin reader */
    struct ReaderData {

        /** The buffer */
        char buffer[READER_DATA_BUFFER_SIZE];

        /** The length of the valid buffer content */
        unsigned int len;

    };

    /** ctor */
    Curser(void) {
        // initialize terminal information database
        setupterm(reinterpret_cast<char *>(0), 1, reinterpret_cast<int *>(0));

        // get number of supported colors (should be 8)
        int i;
        i = tigetnum("colors");
        this->colorsAvailable = (i >= 8);

        // refresh the flag because of the undefined initialization sequence 
        vislib::sys::Console::EnableColors(this->colorsAvailable);

        if (!isatty(STDOUT_FILENO) || !isatty(STDERR_FILENO)) {
            vislib::sys::Console::EnableColors(false);
        }

        // console title crowbar is not initialized.
        this->consoleTitleInit = false;

        this->isXterm = false;
        this->dcopPresent = false;
        this->isKonsole = false;

        // no old console title
        this->oldConsoleTitle = NULL;
    }

    /** dtor */
    ~Curser(void) {
        
        if (this->oldConsoleTitle != NULL) {
            this->SetConsoleTitle(this->oldConsoleTitle);
            ARY_SAFE_DELETE(this->oldConsoleTitle);
        }
    }

    /** getter to colorsAvailable */
    inline bool AreColorsAvailable(void) {
        return this->colorsAvailable;
    }

    /** output function */
    static int outputChar(int c) {
        fputc(c, stdout);
        fputc(c, stderr);
        return c;
    }

    /** wrapper for color setting */
    inline void SetColor(bool foreground, vislib::sys::Console::ColorType col) {
        int colType = COLOR_BLACK;

        // Translate color codes (the hard way, because of the ANSI-constant screw up
        switch (col) {
            case BLACK: colType = COLOR_BLACK; break;
            case DARK_RED: colType = COLOR_RED; break;
            case DARK_GREEN: colType = COLOR_GREEN; break;
            case DARK_YELLOW: colType = COLOR_YELLOW; break;
            case DARK_BLUE: colType = COLOR_BLUE; break;
            case DARK_MAGENTA: colType = COLOR_MAGENTA; break;
            case DARK_CYAN: colType = COLOR_CYAN; break;
            case GRAY: colType = COLOR_WHITE; break;

            case DARK_GRAY: colType = COLOR_BLACK; break;
            case RED: colType = COLOR_RED; break;
            case GREEN: colType = COLOR_GREEN; break;
            case YELLOW: colType = COLOR_YELLOW; break;
            case BLUE: colType = COLOR_BLUE; break;
            case MAGENTA: colType = COLOR_MAGENTA; break;
            case CYAN: colType = COLOR_CYAN; break;
            case WHITE: colType = COLOR_WHITE; break;

            case UNKNOWN_COLOR: 
            default: return; break;
        }

        if (foreground) {
            // color up bright foreground colors using the *BOLD*-Crowbar
            tputs(tparm(set_attributes, 0, 0, 0, 0, 0, (col & 0x08), 0, 0, 0), 1, outputChar);
        }

        tputs(tparm(foreground ? set_a_foreground : set_a_background, colType), 1, outputChar);
    }

    /** helper function to aviod linux stdin read lock */
    static DWORD AskStdin(const void *userData) {
        struct ReaderData *rd = static_cast<struct ReaderData*>(const_cast<void*>(userData));
        rd->len = 0;
        rd->buffer[0] = 0;

        int n;
                    //char s[1024];
        n = read(STDIN_FILENO, rd->buffer, READER_DATA_BUFFER_SIZE - 1);
        rd->buffer[READER_DATA_BUFFER_SIZE - 1] = 0;
        rd->len = n;

        //while(true) {
        //    printf(".");
        //    vislib::sys::Thread::Sleep(500);
        //}

        return 0;
    }

    /** sets the console title, if possible */
    inline void SetConsoleTitle(const char *title) {
        if (!this->consoleTitleInit) {
            // first time call

            { // check capabilities
                vislib::StringA out;
                vislib::StringA err;

                this->dcopPresent = false;
                this->isKonsole = false;

                // first check if dcop is available
                vislib::sys::Console::Run("dcop", &out, &err);

                this->dcopPresent = (err.Length() == 0);

                // check if environment variable $KONSOLE_DCOP_SESSION is present
                char *v = ::getenv("KONSOLE_DCOP_SESSION");

                this->isKonsole = (v != NULL);

                // check if environment variable $TERM is 'xterm'
                v = ::getenv("TERM");

                this->isXterm = ((v != NULL) && (strcasecmp(v, "xterm") == 0));
                if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
                    this->isXterm = false;
                }

            }

            if (this->oldConsoleTitle == NULL) {
                // try to store the old title
                vislib::StringA oldName;

                if (this->dcopPresent && this->isKonsole) {
                    vislib::StringA cmd;
                    cmd.Format("dcop $KONSOLE_DCOP_SESSION sessionName");
                    vislib::sys::Console::Run(cmd.PeekBuffer(), &oldName, NULL);

                } else if (this->isXterm) {
                    // getting title from xterm is very unsecure
                    struct termios tty_ts, tty_ts_orig; // termios settings 
                    struct termios *tty_ts_orig_pt = NULL;

                    // get and backup tty_in termios 
                    tcgetattr(STDIN_FILENO, &tty_ts);
                    tty_ts_orig = tty_ts;
                    tty_ts_orig_pt = &tty_ts_orig;

                    // set tty raw 
                    tty_ts.c_iflag = 0;
                    tty_ts.c_lflag = 0;

                    tty_ts.c_cc[VMIN] = 1;
                    tty_ts.c_cc[VTIME] = 1;
                    tty_ts.c_lflag &= ~(ICANON | ECHO);
                    tcsetattr(STDIN_FILENO, TCSANOW, &tty_ts);

                    printf("\033[21t"); // request title control sequence
                    fflush(stdout);

                    {
                        struct ReaderData rd;
                        rd.len = 0;

                        vislib::sys::Thread stdinreader(AskStdin);
                        stdinreader.Start(&rd);
                        unsigned int cnt = 0;
                        while(cnt < 1000) {
                            vislib::sys::Thread::Sleep(50);
                            cnt += 50;
                            if (rd.len > 0) {
                                break;
                            }
                        }
                        if (rd.len == 0) {                       
                            stdinreader.Terminate(true);
                            rd.len = 0;
                        }

                        if (rd.len > 5) {
                            rd.buffer[rd.len - 2] = 0;
                            oldName = &rd.buffer[3];
                        }

                    }

                    if (tty_ts_orig_pt) {
                        tcsetattr(STDIN_FILENO, TCSAFLUSH, tty_ts_orig_pt);
                    }

                } else { 
                    // another way?

                }

                unsigned int size = oldName.Length();
                if (size > 0) {
                    this->oldConsoleTitle = new char[size + 1];
                    ::memcpy(this->oldConsoleTitle, oldName.PeekBuffer(), size * sizeof(char));
                    this->oldConsoleTitle[size] = 0;

                    // truncate control characters at the end
                    size--;
                    while ((size > 0) && (this->oldConsoleTitle[size] < 0x20)) {
                        this->oldConsoleTitle[size--] = 0;
                    }

                }

            }

            this->consoleTitleInit = true;
        }

        if (this->oldConsoleTitle == NULL) {
            // we won't set a new title if we're not able to recreate the current one
            return;
        }

        if (this->dcopPresent && this->isKonsole) {
            vislib::StringA cmd;
            cmd.Format("dcop $KONSOLE_DCOP_SESSION renameSession '%s'", title);
            vislib::sys::Console::Run(cmd.PeekBuffer(), NULL, NULL);

        } else if (this->isXterm) {
            // xterm operating system command: echo '\033]0;AAAAA\007'
            printf("\033]0;%s\007", title);

        } else {
            // another way? 

        }

    }

private:
    /** flag whether there is color text support */
    bool colorsAvailable;

    /** flag whether the console title mechnisms has been initialized. */
    bool consoleTitleInit;

    /** flag whether dcop is available */
    bool dcopPresent;

    /** flag whether this is an xterm */
    bool isXterm;

    /** flag whether the console is a KDE Konsole */
    bool isKonsole;

    /** old console title stored, which should be restored on exit */
    char *oldConsoleTitle;

};

#undef READER_DATA_BUFFER_SIZE

/*
 * Instance of linux term helper class
 */
vislib::sys::Console::Curser vislib::sys::Console::curser = vislib::sys::Console::Curser();

#endif


/*
 * vislib::sys::Console::Run
 */
int vislib::sys::Console::Run(const char *command, StringA *outStdOut, 
        StringA *outStdErr) {
#ifdef _WIN32
    assert(false);
    return -1;
#else /* _WIN32 */
    // TODO: This whole thing is an extremely large crowbar. I think it is
    // inherently unsafe to read the program output in the current manner.
    // TODO: Could use some of the timeout mechanisms?

    pid_t pid;                      // PID of child process executing 'command'.
    int stdErrPipe[2];              // Pipe descriptors for stderr redirect.
    int stdOutPipe[2];              // Pipe descriptors for stdout redirect.
    int cntRead;                    // # of bytes read from redirect pipe.
    int status;                     // Exit status of 'command'.
    const int BUFFER_SIZE = 128;    // Size of 'buffer'.
    char buffer[BUFFER_SIZE];       // Buffer for reading console.
    
    /* Create two pipes for redirecting the child console output. */
    if (::pipe(stdOutPipe) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }
    if (::pipe(stdErrPipe) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    /* Spawn a new subprocess for running the command. */
    pid = ::fork();
    if (pid < 0) {
        /* Forking failed. */
        throw SystemException(__FILE__, __LINE__);

    } else if (pid == 0) {
        /* Subprocess created, I am in the subprocess now. */

        /* We do not need the read end of the pipe, so close it. */
        ::close(stdOutPipe[0]);
        ::close(stdErrPipe[0]);

        /* Redirect stdout and stderr. */
        if (::dup2(stdOutPipe[1], STDOUT_FILENO) == -1) {
            return -1;
        }

        if (::dup2(stdErrPipe[1], STDERR_FILENO) == -1) {
            return -1;
        }

        /* Replace process image with command to execute. */
        ::execl("/bin/sh", "/bin/sh", "-c", command, static_cast<char *>(NULL));

        /* 
         * If this position is reached, an error occurred as the process image
         * has not successfully been replaced with the command.
         */
        return -1;

    } else {
        /* Subprocess created, I am in parent process. */
        
        /* Close the write end of the pipe, we do not need it. */
        ::close(stdOutPipe[1]);
        ::close(stdErrPipe[1]);

        /* Read the console output, if requested. */
        if (outStdOut != NULL) {
            outStdOut->Clear();

            while ((cntRead = ::read(stdOutPipe[0], buffer, BUFFER_SIZE - 1))
                    > 0) {
                buffer[cntRead] = 0;
                *outStdOut += buffer;

                if (cntRead < BUFFER_SIZE - 1) {
                    break;
                }
            }
        }
        ::close(stdOutPipe[0]);

        if (outStdErr != NULL) {
            outStdErr->Clear();

            while ((cntRead = ::read(stdErrPipe[0], buffer, BUFFER_SIZE - 1))
                    > 0) {
                buffer[cntRead] = 0;
                *outStdErr += buffer;

                if (cntRead < BUFFER_SIZE - 1) {
                    break;
                }
            }
        }
        ::close(stdErrPipe[0]);

        /* Wait for the child to finish. */
        return (::wait(&status) != -1) ? WEXITSTATUS(status) : -1;
    } /* end if (pid < 0) */
#endif /* _WIN32 */
}


/*
 * vislib::sys::Console::Write
 */
void vislib::sys::Console::Write(const char *fmt, ...) {
    va_list argptr;
    
    va_start(argptr, fmt);
    ::vfprintf(stdout, fmt, argptr);
    va_end(argptr);
}


/*
 * vislib::sys::Console::WriteLine
 */
void vislib::sys::Console::WriteLine(const char *fmt, ...) {
    va_list argptr;
    
    va_start(argptr, fmt);
    ::vfprintf(stdout, fmt, argptr);
    ::fprintf(stdout, "\n");
    va_end(argptr);
}


/*
 * vislib::sys::Console::Console
 */
vislib::sys::Console::Console(void) {
    throw UnsupportedOperationException("vislib::sys::Console::Console", 
        __FILE__, __LINE__);
}


/*
 * vislib::sys::Console::~Console
 */
vislib::sys::Console::~Console(void) {
}


/*
 * vislib::sys::Console::ColorsAvailable
 */
bool vislib::sys::Console::ColorsAvailable(void) {
#ifdef _WIN32
    return true;
#else // _WIN32
    return vislib::sys::Console::curser.AreColorsAvailable();
#endif // _WIN32
}


/*
 * vislib::sys::Console::ColorsEnabled
 */
bool vislib::sys::Console::ColorsEnabled(void) {
    return vislib::sys::Console::useColors;
}


/*
 * vislib::sys::Console::EnableColors
 */
void vislib::sys::Console::EnableColors(bool enable) {
    vislib::sys::Console::useColors = enable && vislib::sys::Console::ColorsAvailable();
}

/*
 * vislib::sys::Console::RestoreDefaultColors
 */
void vislib::sys::Console::RestoreDefaultColors(void) {
    if (!vislib::sys::Console::useColors) return;

#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return; // TODO: Inform about error?

    // get current info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return; // TODO: Inform about error?
    
    if (defaultFgcolor != UNKNOWN_COLOR) {
        // clear foreground color bits
        info.wAttributes &= ~(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);

        // set new foreground color bits
        unsigned char col = static_cast<unsigned char>(defaultFgcolor);
        if ((col & 0x01) != 0) info.wAttributes |= FOREGROUND_RED;
        if ((col & 0x02) != 0) info.wAttributes |= FOREGROUND_GREEN;
        if ((col & 0x04) != 0) info.wAttributes |= FOREGROUND_BLUE;
        if ((col & 0x08) != 0) info.wAttributes |= FOREGROUND_INTENSITY;
    }
    
    if (defaultBgcolor != UNKNOWN_COLOR) {
        // clear background color bits
        info.wAttributes &= ~(BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_INTENSITY);

        // set new background color bits
        unsigned char col = static_cast<unsigned char>(defaultBgcolor);
        if ((col & 0x01) != 0) info.wAttributes |= BACKGROUND_RED;
        if ((col & 0x02) != 0) info.wAttributes |= BACKGROUND_GREEN;
        if ((col & 0x04) != 0) info.wAttributes |= BACKGROUND_BLUE;
        if ((col & 0x08) != 0) info.wAttributes |= BACKGROUND_INTENSITY;
    }
    
    // set new attribut flaggs
    SetConsoleTextAttribute(hStdout, info.wAttributes);

#else // _WIN32
    tputs(exit_attribute_mode, 1, vislib::sys::Console::Curser::outputChar);

#endif // _WIN32
}


/*
 * vislib::sys::Console::SetForegroundColor
 */
void vislib::sys::Console::SetForegroundColor(vislib::sys::Console::ColorType fgcolor) {
    if (!vislib::sys::Console::useColors) return;
    if (fgcolor == UNKNOWN_COLOR) return;

#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return; // TODO: Inform about error?

    // get current info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return; // TODO: Inform about error?
    
    // clear bits for foreground color
    info.wAttributes &= ~(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);

    // set new foreground color bits
    unsigned char col = static_cast<unsigned char>(fgcolor);
    if ((col & 0x01) != 0) info.wAttributes |= FOREGROUND_RED;
    if ((col & 0x02) != 0) info.wAttributes |= FOREGROUND_GREEN;
    if ((col & 0x04) != 0) info.wAttributes |= FOREGROUND_BLUE;
    if ((col & 0x08) != 0) info.wAttributes |= FOREGROUND_INTENSITY;
    
    // set new attribut flaggs
    SetConsoleTextAttribute(hStdout, info.wAttributes);

#else // _WIN32
    vislib::sys::Console::curser.SetColor(true, fgcolor);

#endif // _WIN32
}


/*
 * vislib::sys::Console::SetBackgroundColor
 */
void vislib::sys::Console::SetBackgroundColor(vislib::sys::Console::ColorType bgcolor) {
    if (!vislib::sys::Console::useColors) return;
    if (bgcolor == UNKNOWN_COLOR) return;

#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return; // TODO: Inform about error?

    // get current info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return; // TODO: Inform about error?
    
    // clear bits for background color
    info.wAttributes &= ~(BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_INTENSITY);

    // set new background color bits
    unsigned char col = static_cast<unsigned char>(bgcolor);
    if ((col & 0x01) != 0) info.wAttributes |= BACKGROUND_RED;
    if ((col & 0x02) != 0) info.wAttributes |= BACKGROUND_GREEN;
    if ((col & 0x04) != 0) info.wAttributes |= BACKGROUND_BLUE;
    if ((col & 0x08) != 0) info.wAttributes |= BACKGROUND_INTENSITY;
    
    // set new attribut flaggs
    SetConsoleTextAttribute(hStdout, info.wAttributes);

#else // _WIN32
    vislib::sys::Console::curser.SetColor(false, bgcolor);

#endif // _WIN32
}


/*
 * vislib::sys::Console::GetForegroundColor
 */
vislib::sys::Console::ColorType vislib::sys::Console::GetForegroundColor(void) {
    if (!useColors) return UNKNOWN_COLOR;

#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return UNKNOWN_COLOR;

    // get info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return UNKNOWN_COLOR;

    // translate foreground color bits
    unsigned char c = 0;
    if ((info.wAttributes & FOREGROUND_RED) != 0) c += 1;
    if ((info.wAttributes & FOREGROUND_GREEN) != 0) c += 2;
    if ((info.wAttributes & FOREGROUND_BLUE) != 0) c += 4;
    if ((info.wAttributes & FOREGROUND_INTENSITY) != 0) c += 8;

    return static_cast<vislib::sys::Console::ColorType>(c);

#else // _WIN32
    return vislib::sys::Console::UNKNOWN_COLOR;

#endif // _WIN32
}


/*
 * vislib::sys::Console::GetBackgroundColor
 */
vislib::sys::Console::ColorType vislib::sys::Console::GetBackgroundColor(void) {
    if (!useColors) return UNKNOWN_COLOR;

#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return UNKNOWN_COLOR;

    // get info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return UNKNOWN_COLOR;

    // translate background color bits
    unsigned char c = 0;
    if ((info.wAttributes & BACKGROUND_RED) != 0) c += 1;
    if ((info.wAttributes & BACKGROUND_GREEN) != 0) c += 2;
    if ((info.wAttributes & BACKGROUND_BLUE) != 0) c += 4;
    if ((info.wAttributes & BACKGROUND_INTENSITY) != 0) c += 8;

    return static_cast<vislib::sys::Console::ColorType>(c);

#else // _WIN32
    return vislib::sys::Console::UNKNOWN_COLOR;

#endif // _WIN32
}


/*
 * vislib::sys::Console::GetWidth
 */
unsigned int vislib::sys::Console::GetWidth(void) {
#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return 0;

    // get info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return 0;

    return info.srWindow.Right + 1 - info.srWindow.Left;

#else // _WIN32
    int value = tigetnum("cols");
    return (value == -2) ? 0 : value;

#endif // _WIN32
}

 
/*
 * vislib::sys::Console::GetHeight
 */
unsigned int vislib::sys::Console::GetHeight(void) {
#ifdef _WIN32
    // get handle
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    if ((hStdout == NULL) || (hStdout == INVALID_HANDLE_VALUE)) return 0;

    // get info
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (::GetConsoleScreenBufferInfo(hStdout, &info) == 0) return 0;

    return info.srWindow.Bottom + 1 - info.srWindow.Top;

#else // _WIN32
    int value = tigetnum("lines");
    return (value == -2) ? 0 : value;

#endif // _WIN32
}


/*
 * vislib::sys::Console::SetTitle
 */
void vislib::sys::Console::SetTitle(const vislib::StringA& title) {
#ifdef _WIN32
    ::SetConsoleTitleA(title);

#else // _WIN32
    vislib::sys::Console::curser.SetConsoleTitle(title);

#endif // _WIN32
}


/*
 * vislib::sys::Console::SetTitle
 */
void vislib::sys::Console::SetTitle(const vislib::StringW& title) {
#ifdef _WIN32
    ::SetConsoleTitleW(title);

#else // _WIN32
    // we only support ANSI-Strings for Linux consoles.
    vislib::sys::Console::curser.SetConsoleTitle(W2A(title));

#endif // _WIN32
}


/*
 * vislib::sys::Console::SetIcon
 */
void vislib::sys::Console::SetIcon(unsigned int id) {
#ifdef _WIN32
    // Creates an HWND handle for the console window
    HWND console = NULL;
    DynamicFunctionPointer<HWND (*)(void)> getConsoleWindow("kernel32", "GetConsoleWindow");
    if (!getConsoleWindow.IsValid()) return; // function not found. Windows too old.
    console = getConsoleWindow();
    if (console == NULL) return; // no console present

    // Creates an HINSTANCE handle for the current application.
    // 'GetModuleHandleA' creates a HMODULE which should be the same as 
    // HINSTANCE, at least this is the common hope.
    HMODULE instance = ::GetModuleHandleA(NULL);
    if (instance == NULL) return; // no instance handle available ... hmm

    // Load the requested icon ressource
    HICON icon = ::LoadIcon(instance, MAKEINTRESOURCE(id)); 
    if (icon == NULL) return; // icon ressource not found.

    // setting the icon.
#ifndef _WIN64
#pragma warning(disable: 4244)
#endif
    ::SetClassLongPtr(console, GCLP_HICON, reinterpret_cast<LONG_PTR>(icon));
    ::SetClassLongPtr(console, GCLP_HICONSM, reinterpret_cast<LONG_PTR>(icon));
#ifndef _WIN64
#pragma warning(default: 4244)
#endif

#else // _WIN32
    // Linux is stupid

#endif // _WIN32
}
