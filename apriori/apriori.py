#!/bin/python

# apriori.py
# -------------------------
# Winter 2013; Alex Safatli
# -------------------------
# A program that carries out
# the Apriori algorithm on a
# given set of data files.
#
# Usage: python apriori.py

# Constants

RULE_FILE = 'Rules' # Name of Rules file.

# Imports

import os, collections
import itertools as ite

# Apriori Class(es)

class apriori_analysis():
    # The Apriori Algorithm - contains both the itemset
    # finding phase (get_itemsets) and generation of association
    # rules (rule_gen).
    def __init__(self,headers,data,min_sup,min_conf):
        self.headers = headers
        self.database = data
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.candidate_itemsets = collections.defaultdict(list)
        self.frequent_itemsets = collections.defaultdict(list)
        self.supports = {}
        self.rules = []
    
    def get_itemsets(self):
        C = self.candidate_itemsets
        L = self.frequent_itemsets
        self.frequent_1itemsets() # Get 1-item itemsets.
        k = 2
        while len(L[k-1]) > 0: # Search in order.
            self.apriori_gen(k) # Candidate generation.
            indatabase = self.frequent_subset(k)
            L[k] = indatabase
            k += 1
    
    def rule_gen(self):
        # Generate strong association rules.
        
        def has_pruned(se,pr):
            for x in pr:
                if se.issubset(x):
                    return True # Bad rule by pruning. 
            return False
        
        L = self.frequent_itemsets
        sup = self.support
        for k in xrange(2,len(L)):
            # Go through all L with k >= 2.
            for itemset in L[k]:
                # Go through all itemsets in given L.
                pruned, S = [], []
                for l in xrange(1,len(itemset)):
                    # Get all possible subsets.
                    S.extend(list(ite.combinations(itemset,l)))
                for s in S:
                    # For any given subset.
                    se = set(s)
                    if has_pruned(se,pruned):
                        continue # Don't go further.
                    su = sup(itemset)
                    ss = sup(s)
                    if ss != 0:
                        conf = su/ss
                        if (conf >= self.min_conf):
                            # Strong rule found.
                            diff = tuple(set(itemset)-set(s))
                            rul = rule(s,diff,su,conf)
                            if len(diff) > 0 and rul not in self.rules:
                                self.rules.append(rul)
                        else:
                            # Efficiency trick / pruning rule
                            # applied for greater efficiency.
                            pruned.append(se)
    
    def frequent_1itemsets(self):
        # Get 1-item frequent itemsets.
        itemset = self.frequent_itemsets[1]
        for header in self.headers:
            for row in self.database:
                value = row.data[header]
                tupl = (item(header,value),)
                if tupl not in itemset \
                   and self.support(tupl) >= self.min_sup:
                    itemset.append(tupl)
    
    def apriori_gen(self,k):
        # Candidate generation.
        
        def join(i1,i2):
            # Effectively perform a natural
            # join of a set by performing a set
            # union.
            j = set(i1).union(set(i2))
            return tuple(j)                 
        
        itemset = self.candidate_itemsets[k]
        freqset = self.frequent_itemsets[k-1]
        candidates = set() # Ensures no repeats.
        for a in freqset: 
            for b in freqset:
                if a[k-2].header == b[k-2].header:
                    continue # Cannot have identical attributes.
                c = join(a,b) # join for generating candidates
                if (len(c) == k) and \
                   (not self.has_infrequent_subset(c,k-1)):
                    candidates.add(c)
        itemset.extend(list(candidates))
    
    def has_infrequent_subset(self,c,ind):
        # Check infrequent subsets for pruning. Apply
        # apriori knowledge.
        L = self.frequent_itemsets[ind]
        # For each subset.
        subsets = tuple(ite.combinations(c,ind)) # All ind-subsets.
        for s in subsets:
            if s not in L:
                return True
        return False
    
    def frequent_subset(self, k):
        C = self.candidate_itemsets[k]
        c = []
        for itemcol in C:
            if (self.support(itemcol) >= self.min_sup) \
               and (itemcol not in c):
                c.append(itemcol)
        return c
    
    def support(self,itemcol):
        # Get support for a given set of header, value pairs.
        if itemcol in self.supports:
            return self.supports[itemcol]
        count = 0
        for row in self.database:
            of_set = 0
            for item in itemcol:
                header,value = (item.header,item.value)
                if (row.data[header] == value):
                    of_set += 1
                else:
                    break
            if (of_set == len(itemcol)):
                # All items in row.
                count += 1
        sup = count/float(len(self.database))
        self.supports[itemcol] = sup
        return sup   
    
    def __str__(self):
        # Summarizes all results and returns as
        # string.
        out = 'Summary:\nTotal rows in the original set: %d\n' % (len(self.database))
        out += 'Total rules discovered: %d\nThe selected measures: Support=%.2f, Confidence=%.2f\n' % \
            (len(self.rules),self.min_sup,self.min_conf)
        out += '--------------------------------------------------\nRules:\n\n'
        for r in self.rules:
            out += 'Rule#%d: %s\n\n' % (self.rules.index(r)+1,str(r))
        return out

# Helper Class(es) and Function(s)

class item():
    # Encapsulates a header-value pair.
    def __init__(self,header,value):
        self.header = header
        self.value = value
    def __eq__(self,other): # Equality
        return (other.header == self.header \
                and other.value == self.value)
    def __ne__(self,other):
        return not self.__eq__(other)
    def __hash__(self): # Make hashable.
        return id(self)
    def __str__(self): # toString
        return '%s=%s' % (self.header,self.value)

class transaction():
    # Encapsulates a given row or transaction
    # in the dataset.
    def __init__(self,headers,values):
        self.data = {}
        if len(headers) != len(values):
            raise IndexError('Mismatched lengths for headers and values.')
        for header in headers:
            self.data[header] = values[headers.index(header)]

class rule():
    # Encapsulates a generated rule.
    def __init__(self,x,y,sup,conf):
        self.LHS = x
        self.RHS = y
        self.lstr = '%s' % ', '.join(map(str,self.LHS))
        self.rstr = '%s' % ', '.join(map(str,self.RHS))
        self.sup = sup
        self.conf = conf
    def __eq__(self,other):
        return (set(self.LHS) == set(other.LHS)) and \
           (set(self.RHS) == set(other.RHS))
    def __ne__(self,other):
        return not self.__eq__(other)
    def __str__(self):
        return '(Support=%.2f,Confidence=%.2f)\n{ %s } ----> { %s }' \
               % (self.sup,self.conf,self.lstr,self.rstr)

# Main Function

def main():
    
    # Get user input for min_sup, min_conf, and data file.
    fi = raw_input('Specify the path for the data file you wish to analyze: ')
    s = raw_input('Specify the minimum support rate you wish to use for analysis (0.00-1.00): ')
    c = raw_input('Specify the minimum confidence rate you wish to use for analysis (0.00-1.00): ')
    
    # Check input file.
    if not os.path.isfile(fi):
        print '<%s> is not a file. Check the provided path.' % (fi)
        exit()
        
    # Check other parameters.
    try:
        min_sup = float(s)
        min_conf = float(c)
    except:
        print 'Could not parse floating point values for one or both of the given rates.'
        exit()
    if (min_sup > 1.00 or min_sup < 0.00) or (min_conf > 1.00 or min_conf < 0.00):
        print 'Invalid support or confidence rate given (negative or greater than 1.00).'
        exit()
        
    # Get data from file.
    fh = open(fi,'r')
    lines = fh.read().split('\n')
    fh.close()
    values = []
    data = []
    
    # First line are headers/attributes.
    headers = lines[0].split()
    
    # Get remaining lines.
    for i in xrange(1,len(lines)):
        values.append(lines[i].split())
        
    # Convert all found data into transactions.
    for row in values:
        if len(row) == 0: # Empty line.
            continue
        try:
            t = transaction(headers,row)
        except IndexError:
            print '<%s> had a problem being read at line %d. Check data file.' % (fi,values.index(row)+2)
            continue
        data.append(t)
        
    # Perform apriori analysis on file.
    a = apriori_analysis(headers,data,min_sup,min_conf)
    a.get_itemsets() # Candidate generation/tests.
    a.rule_gen() # Generate rules from frequent itemsets L.
    
    # Output results to file.
    fh = open(RULE_FILE,'w')
    fh.write(str(a))
    fh.close()
    
    # Print success message.
    print '\nThe result is in the file %s.\n*** Algorithm Finished ***' \
          % (RULE_FILE)
    
# If not imported.

if __name__ == '__main__':
    main()