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

my $console = getRevisionInfo($basepath . '/Console');
my $viewer = getRevisionInfo($basepath . '/Viewer');
my $glut = getRevisionInfo($basepath . '/Glut');

$hash{'$CONSOLE_REV$'} = $console->rev;
$hash{'$CONSOLE_YEAR$'} = substr($console->date, 0, 4);
$hash{'$CONSOLE_DIRTY$'} = $console->dirty;
$hash{'$VIEWER_REV$'} = $viewer->rev;
$hash{'$VIEWER_DIRTY$'} = $viewer->dirty;
$hash{'$GLUT_REV$'} = $glut->rev;
$hash{'$GLUT_YEAR$'} = substr($glut->date, 0, 4);
$hash{'$GLUT_DIRTY$'} = $glut->dirty;

processFile("consoleversion.gen.h", "consoleversion.template.h", \%hash);
