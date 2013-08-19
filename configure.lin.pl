#!/usr/bin/perl
#
# configure.lin.pl
# MegaMol Core
#
# Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#
use Cwd qw{abs_path};
use strict;
use warnings 'all';
my $incpath = abs_path($0);
$incpath =~ s/\/[^\/]+$//;
push @INC, "$incpath/configperl";
require configperl;

my $fullauto = 0;
if ((grep {$_ eq "fullauto"} @ARGV) || (defined $ENV{'CONFIGPERL_FULLAUTO'})) {
    $fullauto = 1;
}

my ($a, $b, $c);
my @pps = ();
my @fps = ();
my @cfps = ();
my @sps = ();

$a = PathParameter->new();
    $a->id("outbin");
    $a->description("Path to the global \"bin\" output directory");
    $a->placeholder("%outbin%");
    $a->autoDetect(0);
    $a->value("../bin");
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
    $b->value(0);
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

VISUS::configperl::Configure("MegaMol(TM) Core Configuration for Linux", ".megamol.core.lin.cache", \@pps, \@fps, \@cfps, \@sps, $fullauto);

