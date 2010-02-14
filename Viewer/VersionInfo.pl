#
# versioninfo.pl
#
# Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
push @INC, "../rev2ver";
push @INC, "rev2ver";
require "rev2ver.inc";

my $path = shift;

my %hash;

my $versioninfo = getRevisionInfo($path);

$hash{'$VERSION_REV$'} = $versioninfo->rev;
$hash{'$VERSION_YEAR$'} = substr($versioninfo->date, 0, 4);
$hash{'$VERSION_DIRTY$'} = $versioninfo->dirty;

processFile($path . "/include/MegaMolViewer.version.gen.h", $path . "/version.template.h", \%hash);
