#
# genFiles.pl
#
# Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
# Alle Rechte vorbehalten.
#
use File::Basename;
$#ARGV == 1 || die "Uhrgmlnhagi";

my $apiHeader = $ARGV[0];
my $impHeader = $ARGV[1];
my $header = "";


#
# Reading input
#
open (INAPIHEADER, $apiHeader) || die "Cannot open Core API Header \"$apiHeader\"";
while (<INAPIHEADER>) {
    chomp;
    $header .= " " . $_;
}
close (INAPIHEADER);

$header =~ s/\*/ \* /g;
$header =~ s/\s/ /g;
$header =~ tr/ / /s;


#
# Parsing input
#

# MEGAMOLCORE_API mmcErrorCode MEGAMOLCORE_CALL mmcSetInitialisationValue(
#     void *hCore, mmcInitValueEnum key, mmcValueType type, const void* value);

my @functions;

while ($header =~ m/MEGAMOLCORE_API\s+([^\(#]+)\s+MEGAMOLCORE_CALL\s+([^\(]+)\(\s*([^\)]*)\s*\)/g) {
    my @match;
    push @match, $1;
    push @match, $2;
    push @match, $3;
    push @functions, [@match];
}


#
# Specify functions to import
#
my @funcsToUse = (
    "mmcGetVersionInfo",
    "mmcGetHandleSize",
    "mmcDisposeHandle",
    "mmcCreateCore",
    "mmcIsHandleValid",
    "mmcSetInitialisationValue",
    "mmcInitialiseCoreInstance",
    "mmcRequestInstanceA",
    "mmcRequestInstanceW",
    "mmcHasPendingViewInstantiationRequests",
    "mmcInstantiatePendingView",
    "mmcRenderView",
    "mmcResizeView",
);


#
# Writing output
#
open (OUTFILE, ">$impHeader") || die "Cannot write Core API import file \"$impHeader\"";

print OUTFILE qq§
///
/// MMCoreApi.def.cs
///
/// Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
/// Alle Rechte vorbehalten.
///
/// This file is generated. Do not change!
///
§;

print OUTFILE qq§
using System;
using System.Runtime.InteropServices;

namespace MegaMol.Core {

    partial class MegaMolCore {
§;

for $func ( @functions ) {
    my $retval = @$func[0];
    my $name = @$func[1];
    my $params = @$func[2];

    if (!grep $_ eq $name, @funcsToUse) {
        print OUTFILE "        // $retval $name($params);\n";
        next;
    }

    $params =~ s/const void \*/IntPtr/g;
    $params =~ s/mmcOSys \*/ref mmcOSys/g;
    $params =~ s/mmcHArch \*/ref mmcHArch/g;
    $params =~ s/bool \*/ref bool/g;
    $params =~ s/unsigned int \*/ref uint/g;
    $params =~ s/int \*/ref int/g;
    $params =~ s/void \*/IntPtr/g;
    $params =~ s/unsigned short \*/ref UInt16/g;
    $params =~ s/unsigned short/UInt16/g;
    $retval =~ s/unsigned int/uint/g;
    $params =~ s/unsigned int/uint/g;
    $params =~ s/void//g;
    $params =~ s/const char \*/[MarshalAs(UnmanagedType.LPStr)]String/g;
    $params =~ s/const wchar_t \*/[MarshalAs(UnmanagedType.LPWStr)]String/g;
    $params =~ s/char \*/[MarshalAs(UnmanagedType.LPStr)]StringBuilder/g;
    $params =~ s/wchar_t \*/[MarshalAs(UnmanagedType.LPWStr)]StringBuilder/g;

    print OUTFILE qq§        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate $retval $name§ . qq§Delegate($params);
        public $name§ . qq§Delegate $name = null;
§;
}

print OUTFILE qq§
    }
}
§;

close (OUTFILE);

#for $func ( @functions ) {
#    print "# " . @$func[0] . " # " . @$func[1] . " # " . @$func[2] . " #\n";
#}
# print join("\n::::  ", @functions);
#print "\n\nHello World from $projDir\n";

exit 0;
