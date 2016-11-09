/* *********************************************************************
 *
 * 
 * 
 * 
 * *********************************************************************
 */
 
 
#include "Python.h"
 
#include <iostream>
#include <cstdlib> // for: mbstowcs() - multi byte string to wide char string
#include <string>


using namespace std;


int main(int argc, char** argv) {
    
	cout << "--- Start of program ---" << endl;
        
    FILE *ScriptFile;
    const char PythonArg0_Script[] = "../UncertaintyInputData.py";
    const char PythonArg1_PDB[]    = "1aon";
    const char PythonArg2_d[]      = "-d";  
         
    wchar_t wPythonArg0[strlen(PythonArg0_Script)];
    wchar_t wPythonArg1[strlen(PythonArg1_PDB)];
    wchar_t wPythonArg2[strlen(PythonArg2_d)];
    mbstowcs(&wPythonArg0[0], &PythonArg0_Script[0], sizeof(wPythonArg0));
    mbstowcs(&wPythonArg1[0], &PythonArg1_PDB[0],    sizeof(wPythonArg1));
    mbstowcs(&wPythonArg2[0], &PythonArg2_d[0],      sizeof(wPythonArg2));

    wchar_t* wPythonArgv[] = {&wPythonArg0[0], &wPythonArg1[0], &wPythonArg2[0], NULL};
    int      wPythonArgc   = (int)(sizeof(wPythonArgv) / sizeof(wPythonArgv[0])) - 1;
    
    
    
    // initialize the embedded python interpreter
    Py_SetProgramName(wPythonArgv[0]);
    Py_Initialize();
    PySys_SetArgv(wPythonArgc, wPythonArgv);


    // open script file
    ScriptFile = fopen(PythonArg0_Script, "r");
    if(ScriptFile != NULL) {
        // call python script with interpreter
        PyRun_SimpleFileEx(ScriptFile, PythonArg0_Script, 1); // last parameter == 1 means to close the file before returning.
        // DON'T call: fclose(ScriptFile);

    }
    else {
        cout << ">>> ERROR: Couldn't find/open file: \"" << PythonArg0_Script << "\"" << endl;
    }


    // end the python interpreter
    Py_Finalize();
    
	cout << "--- End of program ---" << endl;
	return 0;
}
