/*
 * [!output PROJECT_NAME].cpp  [!output CURRENT_DATE]
 *
 * Copyright (C) [!output CURRENT_YEAR]
 */

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#endif /* _WIN32 */


#ifdef _WIN32
/**
 * Entry point of the application.
 *
 *
 * @param hInstance     Handle to the current instance of the application. 
 * @param hPrevInstance Handle to the previous instance of the application. This
 *                      parameter is always NULL.
 * @param lpCmdLine     Pointer to a null-terminated string specifying the 
 *                      command line for the application, excluding the program 
 *                      name.
 * @param nCmdShow      Specifies how the window is to be shown.
 *
 * @return If the function succeeds, terminating when it receives a WM_QUIT 
 *         message, it should return the exit value contained in that message's 
 *         wParam parameter. If the function terminates before entering the 
 *         message loop, it should return zero. 
 */
int APIENTRY _tWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, 
        LPTSTR lpCmdLine, int nCmdShow) {
#else /* _WIN32 */
/**
 * Entry point of the application.
 *
 * @param argc An integer that contains the count of arguments that follow in 
 *             'argv'. The 'argc' parameter is always greater than or equal 
 *             to 1.
 * @param argv An array of null-terminated strings representing command-line 
 *             arguments entered by the user of the program. By convention, 
 *             argv[0] is the command with which the program is invoked, argv[1]
 *             is the first command-line argument, and so on.
 *
 * @return Application-specific return value.
 */
int main(int argc, char **argv) {
#endif /* _WIN32 */
    return 0;
}
