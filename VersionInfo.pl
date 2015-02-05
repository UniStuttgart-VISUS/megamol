#
# versioninfo.pl
#
# Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
push @INC, "rev2ver";
require "rev2ver.inc";

my $basepath = shift;

my %hash;

my $core = getRevisionInfo($basepath . '/.');
my $coreapi = getRevisionInfo($basepath . '/include/mmcore/api');

$hash{'$CORE_REV$'} = $core->rev;
$hash{'$CORE_YEAR$'} = substr($core->date, 0, 4);
$hash{'$CORE_DIRTY$'} = $core->dirty;
$hash{'$CORE_URI$'} = $core->uri;
$hash{'$COREAPI_REV$'} = $coreapi->rev;
$hash{'$COREAPI_DIRTY$'} = $coreapi->dirty;

processFile("./include/mmcore/productversion.gen.h", "./include/mmcore/productversion.template.h", \%hash);
processFile("./include/mmcore/api/MegaMolCore.std.h", "./include/mmcore/api/MegaMolCore.std.template.h", \%hash);
