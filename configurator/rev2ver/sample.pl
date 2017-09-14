#
# sample.pl
# Copyright (c) 2009, VISUS
# (Visualization Research Center, Universitaet Stuttgart)
# All rights reserved.
#
# See: LICENCE.TXT or
# https://svn.vis.uni-stuttgart.de/utilities/rev2ver/LICENCE.TXT
#
push @INC, "rev2ver";
require "rev2ver.inc";

my $path = ".";
my %hash;

my $stuff = getRevisionInfo($path);
$hash{'$HUGO$'} = $stuff->rev;
$hash{'$EGON$'} = $stuff->uri;
$hash{'$HORST$'} = $stuff->date;
$hash{'$GODZILLA$'} = $stuff->author;
$hash{'$MOTHRA$'} = $stuff->dirty;
$hash{'$GAMERA$'} = $stuff->modDate;
processFile("ipsum.txt", "lorem.txt", \%hash);
