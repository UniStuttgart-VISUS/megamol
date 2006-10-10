#
# createclass.ps1
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

# Greeting
echo ""
echo "    VISlib CreateClass powershell script"
echo "    Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten."
echo ""

# Check arguments
if ($Args.Length -eq 3) {
	$Project = $Args[0]
	$Mode = $Args[1]
	$ClassName = $Args[2]
} else {
	echo "Usage: createclass.ps1 <project> [public|private] <classname>"
    echo ""
	exit
}

# Check creation mode and set path variables
$CPPPath = ".\$Project\src\"
if ($Mode -eq "public") {
    $HPath = ".\$Project\include\vislib\"
} elseif ($Mode -eq "private") {
    $HPath = ".\$Project\src\"
} else {
    echo "Error: You must specify either `"public`" or `"private`" as creation mode."
    echo ""
    exit
}

# Check Project
if (! (test-path $CPPPath)) {
    echo "Error: No project named `"$Project`" found."
    echo ""
    exit
}

# Make file names with path
$CPPFile = $CPPPath + $ClassName + ".cpp"
$HFile = $HPath + $ClassName + ".h"

# Check if classes already exists
if ((test-path $CPPFile) -or (test-path $HFile)) {
    echo "Error: Project `"$Project`" has already a class named `"$ClassName`"."
    echo ""
    exit
}

# Summary and confirmation
echo "Creating $Mode class `"$ClassName`" in project `"$Project`"";
$confirmation = read-host "Press 'm' to make it so"
if ($confirmation -ne "m") {
    echo ""
    exit
}

# Writing CPP file
$CPPContent = "/*
 * " + $ClassName + ".cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include `""

if ($Mode -eq "public") {
    $CPPContent += "vislib/"
}

$CPPContent += $ClassName + ".h`"


/*
 * vislib::" + $ClassName + "::" + $ClassName + "
 */
vislib::" + $ClassName + "::" + $ClassName + "(void) {
    // TODO: Implement
}


/*
 * vislib::" + $ClassName + "::~" + $ClassName + "
 */
vislib::" + $ClassName + "::~" + $ClassName + "(void) {
    // TODO: Implement
}"

# Writing H file
$HContent = "/*
 * " + "$ClassName" + ".h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_" 
$HContent += $ClassName.ToUpper()
$HContent += "_H_INCLUDED
#define VISLIB_"
$HContent += $ClassName.ToUpper()
$HContent += "_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


namespace vislib {
/** TODO: Add additional Namespaces here */


    /**
     * TODO: comment class
     */
    class " + $ClassName + " {

    public:

        /** ctor */
        " + $ClassName + "(void);

        /** Dtor. */
        ~" + $ClassName + "(void);

    protected:

    private:

    };
    
} /* end namespace vislib */

#endif /* VISLIB_" + $ClassName + "_H_INCLUDED */"

#echo $HContent > $HFile
ni . -name $HFile -force -type "file" -value ($HContent | out-string) >> $null
ni . -name $CPPFile -force -type "file" -value ($CPPContent | out-string) >> $null

echo "Done."
echo ""
