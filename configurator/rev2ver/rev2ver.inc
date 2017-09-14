#
# rev2ver.pl
# Copyright (c) 2009,2015 VISUS
# (Visualization Research Center, Universitaet Stuttgart)
# All rights reserved.
#
# See: LICENCE.TXT or
# https://svn.vis.uni-stuttgart.de/utilities/rev2ver/LICENCE.TXT
#
use strict;
use Class::Struct;
use File::Spec::Functions qw{rel2abs};
use File::stat;
use IO::Dir;
use SvnWrapper;
use RelevanceChecker;

my $svnOLE;
my $svnwrap;
our $theLocalPath;
our $theRemotePath;
our $DEBUGPR = 0;

struct revisionInfo => [
    rev => '$',
    author => '$',
    dirty => '$',
    uri => '$',
    date => '$',
    modDate => '$'
];

# getRevisionInfo(path)
# extract information from the working copy at path
sub getRevisionInfo($) {
    my $path = rel2abs(shift);
    $theLocalPath = $path;
    my $rev = 0;
    my $author;
    my $dirty = 0;
    my $uri;
    my $date;
    my $newest = 0;
    my @modifiedFiles = ();
    my %localDefaultFilter = ( default => "consider" );
    my $filterRef = shift;
    if (! $filterRef) {
        $filterRef = \%localDefaultFilter;
    }

    $svnwrap = SvnWrapper->new($filterRef);
    $svnwrap->SetPath($path);

    # this should go and be replaced by $svnwrap->RelevantRev
    foreach my $x (@{$svnwrap->RevPerFile}) {
        my @arr = @{$x};
        #print "$arr[0]: $arr[1]\n";
        if (RelevanceChecker::checkPathRelevance($arr[0], $filterRef)) {
            if ($arr[1] > $rev) {
                #print "rev2ver: setting revision from " . $arr[0] . ": " . $arr[1] . "\n";
                $rev = $arr[1];
            }
        }
    }
    $author = $svnwrap->RevisionInfo->{$rev}[0];
    $date = $svnwrap->RevisionInfo->{$rev}[1];
    
    $uri = $svnwrap->Url;
    $theRemotePath = $uri;

    my $ret = revisionInfo->new();

    # this should go and be replaced by $svnwrap->NewestModification
    if ($svnwrap->HasModifications) {
        foreach my $f (@{$svnwrap->ChangedFiles}) {
            if (RelevanceChecker::checkPathRelevance($f, $filterRef)) {
                $dirty = 1;
                my $mtime = (stat $path."/".$f)->[9];
                if ($mtime > $newest) {
                    #print "rev2ver: setting date from " . ($path."/".$f) . "\n";
                    $newest = $mtime;
                }
            }
        }
        $ret->modDate($newest);
    }

    print "relevantRev: " . $svnwrap->RelevantRev . " rev: " . $rev . "\n";
    print "NewestModification: " . $svnwrap->NewestModification . " newest: " . $newest . "\n";

    $ret->rev($rev);
    $ret->author($author);
    $ret->dirty($dirty);
    $ret->uri($uri);
    $ret->date($date);
    #$DEBUGPR and print "GRAND RESULT: revision $rev and dirtyness $dirty\n";
    print "$uri revision $rev by $author from $date, dirtyness $dirty\n";
    return $ret;
}

# processFile(outFile, inFile, keywordMap)
# load inFile
# foreach k (keys keywordMap) {
#   replace all occurrences of k with keywordMap{k}
# }
# write outFile
sub processFile($$%) {
    my $outFile = shift;
    my $inFile = shift;
    my %hash = %{shift()};
    my $k;

    my $outputBuffer = "";

    if (open IN, $inFile) {
        while(<IN>) {
            foreach $k (keys %hash) {
                #print $k . " -> " . $hash{$k} . "\n";
                s/\Q$k\E/$hash{$k}/g;
            }
            $outputBuffer .= $_;
        }
        close IN;
    } else {
        print "could not read from " . $inFile . "\n";
    }

    writeFileChanges($outFile, $outputBuffer);
}

# writeFileChanges(outFile, content)
# write content to outFile only if the content is changed
sub writeFileChanges($$) {
    my $outFile = shift;
    my $outputBuffer = shift;
    
    if (open IN, $outFile) {
        undef $/;

        my $old = <IN>;
        close IN;

        if ($old eq $outputBuffer) {
            return;
        }
    }

    if (open OUT, ">".$outFile) {
        print OUT $outputBuffer;
        close OUT;
    } else {
        print "could not write to " . $outFile . "\n";
    }
}

return 1;