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

#else /* _WIN32 */
#include <stdlib.h>

#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <curses.h>
#include <term.h>

#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


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
    }

    /** dtor */
    ~Curser(void) {
    }

    /** getter to colorsAvailable */
    inline bool AreColorsAvailable(void) {
        return this->colorsAvailable;
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
            putp(tparm(set_attributes, 0, 0, 0, 0, 0, (col & 0x08), 0, 0, 0));
        }

        putp(tparm(foreground ? set_a_foreground : set_a_background, colType));
    }

private:
    /** flag whether there is color text support */
    bool colorsAvailable;

};


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
    throw UnsupportedOperationException("vislib::sys::Console::Ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::Console::~Console
 */
vislib::sys::Console::~Console(void) {
    // TODO: Implement
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
    putp(exit_attribute_mode);

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
