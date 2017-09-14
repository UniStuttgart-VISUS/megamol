package RelevanceChecker;
use strict;

sub checkPathRelevance {
    my ($file, $filterRef, $DEBUGPRINT) = @_;
    if (!defined $DEBUGPRINT) {
        $DEBUGPRINT = 0;
    }
    
    my @ignore = ();
    my @consider = ();
    
    if ($filterRef->{ignore}) {
        @ignore = @{$filterRef->{ignore}};
    }
    if ($filterRef->{consider}) {
        @consider = @{$filterRef->{consider}};
    }
        
    if (defined scalar @ignore) {
        my $t = 0;
        foreach my $i (@ignore) {
            $DEBUGPRINT and print "ignore: checking $file against $i: ";
            if ($file =~ qr/$i/) {
                $DEBUGPRINT and print "OK.\n";
                $t++;
                last;
            } else {
                $DEBUGPRINT and print "nope.\n";
            }
        }
        if ($t gt 0) {
            $DEBUGPRINT and print "----> $file is ignored.\n";
            return 0;
        }
    }
    if (defined scalar @consider) {
        my $t = 0;
        foreach my $i (@consider) {
            $DEBUGPRINT and print "consider: checking $file against $i: ";
            if ($file =~ qr/$i/) {
                $DEBUGPRINT and print "OK.\n";
                $t++;
                last;
            } else {
                $DEBUGPRINT and print "nope.\n";
            }
        }
        if ($t gt 0) {
            $DEBUGPRINT and print "----> $file is considered.\n";
            return 1;
        }
    }
    if ($filterRef->{default} eq "consider") {
        $DEBUGPRINT and print "----> $file is considered by default.\n";
        return 1;
    } else {
        $DEBUGPRINT and print "----> $file is ignored by default.\n";
        return 0;
    }
}

return 1;