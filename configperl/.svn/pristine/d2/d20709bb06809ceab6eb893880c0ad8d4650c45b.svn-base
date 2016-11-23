#!/usr/bin/perl
#
# config_template.pl
# Copyright (c) 2009, VISUS
# (Visualization Research Center, Universitaet Stuttgart)
# All rights reserved.
#
# See: LICENCE.TXT or
# https://svn.vis.uni-stuttgart.de/utilities/configperl/LICENCE.TXT
#
use Cwd qw{abs_path};
use strict;
use warnings 'all';
my $incpath = abs_path($0);
$incpath =~ s/\/[^\/]+$//;
push @INC, "$incpath";
require configperl;

my @pps = ();
my @fps = ();
my @cfps = ();
my @sps = ();

my $a = PathParameter->new();
$a->id("horst");
$a->description("wir machen uns zum horst fuer visus");
$a->placeholder("HORST");
$a->markerFile("GL/glut.h");
$a->relativeLocation("..");
$a->autoDetect(1);
#$a->directorySeparator("\\");
$a->enforceTrailingDirectorySeparator(1);
push @pps, $a;

my $z = PathParameter->new();
$z->id("hugo");
$z->description("die dummen machen uns zum hugo fuer visus");
$z->placeholder("HuGo");
$z->markerFile("GL/glut.h");
$z->relativeLocation("..");
$z->autoDetect(1);
$z->dependencyFlagID("bestupid");
$z->dependencyDisabledValue("ARGHL-NOT-STUPID");
push @pps, $z;

my $q = PathParameter->new();
$q->id("heinz");
$q->description("die dummen machen uns zum heinz fuer visus");
$q->placeholder("HuGo");
$q->markerFile("GL/glut.h");
$q->relativeLocation("..");
$q->autoDetect(1);
$q->dependencyFlagID("bestupid");
$q->dependencyDisabledValue("ARGHL-NOT-STUPID");
push @pps, $q;

my $b = FlagParameter->new();
$b->id("bestupid");
$b->placeholder("bestupid");
$b->description("be stupid or not, just choose");
$b->value(1);
push @fps, $b;

my $c = ConfigFilePair->new();
$c->inFile("egon.in");
$c->outFile("egon.out");
push @cfps, $c;

my $d = StringPair->new();
$d->placeholder("UDO");
$d->value("juergen");
push @sps, $d;

VISUS::configperl::AddSearchFolder("C:/Windows");

VISUS::configperl::Configure("Konfiguriere den Egon", "egon.cache", \@pps, \@fps, \@cfps, \@sps, \@ARGV);
# use this variant for full auto mode
#VISUS::configperl::Configure("Konfiguriere den Egon", "egon.cache", \@pps, \@fps, \@cfps, \@sps, 1);

