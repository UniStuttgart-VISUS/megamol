#!/usr/bin/perl
#
# configure.win.pl
# VISlib
#
# Copyright (C) 2008-2014 by Universitaet Stuttgart (VISUS) and TU Dresden (CGV)
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
    $a->id("thelib");
    $a->description("Path to thelib++ directory");
    $a->placeholder("%thelib%");
    $a->markerFile("thelib\+\+.vcxproj\$");
    $a->relativeLocation("./");
    $a->autoDetect(1);
    $a->directorySeparator("\\");
    $a->enforceTrailingDirectorySeparator(1);
    push @pps, $a;

$c = ConfigFilePair->new();
    $c->inFile("ExtLibs.props.input");
    $c->outFile("ExtLibs.props");
    push @cfps, $c;

VISUS::configperl::Configure("The.VISlib.Legacy Configuration for Windows", ".the.vislib.legacy.win.cache", \@pps, \@fps, \@cfps, \@sps, $fullauto);

