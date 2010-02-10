#
# configure.lin.pl
# MegaMol Core
#
# Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
push @INC, "configperl";
require "configperl.inc";

my @pps = ();
my @fps = ();
my @cfps = ();
my @sps = ();

$a = PathParameter->new();
    $a->id("outbin");
    $a->description("Path to the global \"bin\" output directory");
    $a->placeholder("%outbin%");
    $a->autoDetect(0);
    push @pps, $a;

$a = PathParameter->new();
    $a->id("vislib");
    $a->description("Path to the vislib directory");
    $a->placeholder("%vislib%");
    $a->markerFile("vislib.sln\$");
    $a->relativeLocation("./");
    $a->autoDetect(1);
    push @pps, $a;
$b = FlagParameter->new();
    $b->id("withUserExpat");
    $b->description("Use a local user compiled expat 2 library (MegaMol-Lib-Naming required)");
    $b->placeholder("%withUserExpat%");
    $b->value(1);
    push @fps, $b;
$a = PathParameter->new();
    $a->id("expat");
    $a->description("Path to the expat directory");
    $a->placeholder("%expatPath%");
    $a->markerFile("expat.h\$");
    $a->relativeLocation("../");
    $a->autoDetect(1);
    $a->dependencyFlagID("withUserExpat");
    $a->dependencyDisabledValue("");
    push @pps, $a;
# libpng is always part of Linux OS
# zlib is always part of Linux OS

$c = ConfigFilePair->new();
    $c->inFile("ExtLibs.mk.input");
    $c->outFile("ExtLibs.mk");
    push @cfps, $c;

Configure("MegaMol(TM) Core Configuration for Linux", ".megamol.core.lin.cache", \@pps, \@fps, \@cfps, \@sps);

