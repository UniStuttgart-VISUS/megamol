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
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/SystemException.h"




/*
 * Coloring under windows with: 

    if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo)) 
    {
        MessageBox(NULL, TEXT("GetConsoleScreenBufferInfo"), 
            TEXT("Console Error"), MB_OK); 
        return 0;
    }

    wOldColorAttrs = csbiInfo.wAttributes; 

    // Set the text attributes to draw red text on black background. 

    if (! SetConsoleTextAttribute(hStdout, FOREGROUND_RED | 
            FOREGROUND_INTENSITY))
    {
        MessageBox(NULL, TEXT("SetConsoleTextAttribute"), 
            TEXT("Console Error"), MB_OK);
        return 0;
    }


	SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN);
	printf("weiss ");
	SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
	printf("weisser ");
	SetConsoleTextAttribute(hStdout, FOREGROUND_RED);
	printf("rot ");
	SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
	printf("roter ");
	SetConsoleTextAttribute(hStdout, BACKGROUND_BLUE | BACKGROUND_RED | BACKGROUND_GREEN);
	printf("andersrum\n");

	//SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN);
	SetConsoleTextAttribute(hStdout, wOldColorAttrs);

 *
 */


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
    // TODO: Implement
}


/*
 * vislib::sys::Console::~Console
 */
vislib::sys::Console::~Console(void) {
    // TODO: Implement
}
