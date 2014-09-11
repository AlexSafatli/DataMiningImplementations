#!/bin/python

# id3.py
# -------------------------
# Winter 2013; Alex Safatli
# -------------------------
# A program that carries out
# the ID3 algorithm on a
# given data file.
#
# Usage: python id3.py

# Imports

import os, sys
from math import log

# Constants

RULES_FILE = 'RULES'

# ID3 Class(es)

class id3:
    # Decision tree ID3 algorithm.
    
    def __init__(self,target,dataset):
        # Tree object.
        self.tree = tree(None,target)
        # Target and dataset.
        self.target, self.dataset = target, dataset
        # The dataset list and entropy for the target.
        self.classes_target = dataset[target]
        self.entropy_target = self.entropy(target)
        
    def entropy(self,attr,subset=None,\
                drange=None):
        out, dataset = 0, self.dataset
        # Get the lists of values.
        tarvals = self.classes_target # target
        tarset = list(set(tarvals))
        attvals = dataset[attr] # attribute
        # Determine what range of indices to look over.
        if not drange:
            drange = range(len(attvals))
        # Set up a count for all unique values in target.
        tarcnt, attcnt = [0 for x in tarset], [0 for x in tarset]
        # For each value in the attribute.
        for i in drange:
            attval = attvals[i]
            # For each unique value in target.
            if (subset and attval == subset) or not subset:
                for j in xrange(len(tarset)):
                    attcnt[j] += 1
                    # See if matching.
                    if tarvals[i] == tarset[j]:
                        tarcnt[j] += 1
        # Log computations.
        for j in xrange(len(tarset)):
            # Add to value.
            if attcnt[j] != 0:
                p = (float(tarcnt[j])/attcnt[j])
                if p != 0: # Avoid finding the log of 0.
                    out -= p*log(p,2)
        return out
    
    def gain(self,attr,drange=None):
        # The attribute given partitions the
        # set into subsets.
        raw = self.dataset[attr]
        data = []
        if drange:
            for d in drange:
                data.append(raw[d])
        else:
            data = raw
        out = self.entropy_target
        subs = set(data)
        for subset in subs:
            # Get ratio of number of occurences
            # of subset in the full set.
            s_j = data.count(subset)
            ratio = float(s_j)/len(data)
            out -= ratio*self.entropy(attr,subset,drange)
        return out
    
    def bestDecision(self,parent,drange=None):
        # Choose best attribute for next decision
        # in the tree.
        bestsplit = None
        if not parent:
            attrs = []
        else:
            attrs = [x for x in parent.getParents()]
        # Check only the given subset of the dataset.
        for header in self.dataset:
            if (header != self.target and \
                header not in attrs):
                gain = self.gain(header,drange)
                # See if better than current best gain.
                if not bestsplit or gain > bestsplit[1]:
                    bestsplit = (header,gain)
        # Split for most gain in information.
        if bestsplit:
            return bestsplit[0]
        else:
            return None # stopping condition reached.
    
    def buildTree(self,parent=None,data=None):
        
        def attachBranches(vals,node):
            # Attach a set of branches to an
            # attrib ute node.
            br = None
            for s in set(vals):
                if br:
                    b = branch(s,None,br,node)
                    br = b
                else:
                    br = branch(s,None,None,node)
            return br
        
        # Figure out what subset of the data is
        # being investigated for this branch.
        attr = self.bestDecision(parent,data)
        if not data:
            data = range(len(self.classes_target))        
        if not attr:
            # If stopping condition reached:
            # no attributes are left or no data.
            # Use class distribution.
            cldist = []
            for d in data:
                cldist.append(self.classes_target[d])
            cla = max(cldist)
            node = attribute(cla,None,parent) # leaf node
            parent.attr = node
            return
        vals = self.dataset[attr]
        # Create branch nodes for new attribute node.
        node = attribute(attr,None,parent)
        br = attachBranches(vals,node)
        node.branch = br
        # Attach new attribute node to parent.
        if parent:
            parent.attr = node # prev branch
        else:
            self.tree = tree(node,self.target) # root node          
        # Recurse.
        majorityclass = []
        for b in br: # Go through all branches.
            # See if all samples for this node have same
            # class label.
            datasubset = [x for x in xrange(len(vals)) \
                          if vals[x] == b.value and x in data]            
            classsubset = []
            for x in datasubset:
                classsubset.append(self.classes_target[x])
            if len(set(classsubset)) == 1:
                # If stopping condition reached. All same class.
                cla = classsubset[0]
                node = attribute(cla,None,b) # leaf node
                b.attr = node
                majorityclass.append(cla)
                continue
            elif len(set(classsubset)) == 0:
                # No class; means not majority class.
                cla = None
                for cl in self.classes_target:
                    # Allows for support for non-binary target 
                    # attributes.
                    if cl not in majorityclass:
                        cla = cl
                        break
                node = attribute(cla,None,b) # leaf node
                b.attr = node
                continue
            self.buildTree(b,datasubset)

# Tree Data Structure

class tree:
    # Encapsulates a tree.
    def __init__(self,attr,target):
        self.root = attr
        self.target = target
    def __nextlevel__(self,node,lvl,out):
        # For an attribute.
        branch = node.branch
        if branch:
            # not leaf node
            for b in branch:
                # For every branch off attribute.
                out.append('\n%sIf %s is %s, then' \
                        % ('\t'*lvl,node.name,b.value))
                self.__nextlevel__(b.attr,lvl+1,out)
        else:
            # leaf node
            out.append(' %s is %s.' % (self.target,node.name))
    def toStringAsLogic(self):
        # Gets string of the tree as if-logic.
        out = []
        self.__nextlevel__(self.root,0,out)
        return "".join(out)

class attribute:
    # An attribute node. A leaf node is 
    # designated as an attribute node 
    # with a null pointer.
    def __init__(self,name,branch,parent):
        self.name = name
        self.branch = branch
        self.parent = parent
    def __eq__(self,other):
        return self.name == other
    def __ne__(self,other):
        return not self.__eq__(other)
    def __str__(self):
        return self.name

class branch:
    # A value branch.
    def __init__(self,value,attr,next,parent):
        self.value = value
        self.attr = attr
        self.next = next
        self.parent = parent
    def getParents(self):
        n = self
        while (n.parent):
            n = n.parent
            if hasattr(n,'attr'): # if branch
                continue
            yield n.name
    def __iter__(self):
        n = self
        while (n):
            yield n
            n = n.next
    def __str__(self):
        return self.value

# Helper Class(es) and Function(s)

def loadFile(fi):
    # Load a file and generate data.    

    # Get data from file.
    fh = open(fi,'r')
    lines = fh.read().split('\n')
    fh.close()
    values = []
    data = {}
    
    # First line are headers/attributes.
    headers = lines[0].split()
    
    # Get remaining lines.
    for i in xrange(1,len(lines)):
        values.append(lines[i].split())
        
    # Convert all found lines to hash table.
    for row in values:
        # Not empty line and correct number of entries.
        if len(row) == len(headers): 
            for header in headers:
                if header not in data:
                    data[header] = []
                data[header].append(row[headers.index(header)])
        # Not empty line, wrong number of entries.
        elif len(row) != len(headers) and len(row) != 0: 
            print '<%s> had a problem being read at line %d. Check data file.' \
                  % (fi,values.index(row)+2)
    
    return data

def askForTarget(dataset):
    # Get target attribute from keyboard input.
    
    def ask(leng):
        val = raw_input('Attribute: ')
        try:
            val = int(val)
        except:
            print 'You did not enter a number.'
            val = ask(leng)
        if (val <= 0) or (val > leng):
            print 'List index out of range. Try again.'
            val = ask(leng)
        return val
    
    # Get all binary keys.
    binarykeys = [x for x in dataset if len(set(dataset[x]))==2]
    # Print as a list to the user.
    print 'Please choose an attribute (by number):'
    for key in binarykeys:
        print '\t%d. %s' % (binarykeys.index(key)+1,key)
    # Get list index + 1.
    val = ask(len(binarykeys))
    return binarykeys[val-1]

# Main Function

def main():
    
    # Get user input for min_sup, min_conf, and data file.
    fi = raw_input('Specify the path for the data file you wish to analyze: ')
    
    # Check input file.
    if not os.path.isfile(fi):
        print '<%s> is not a file. Check the provided path.' % (fi)
        exit()
    
    # Get dataset.
    data = loadFile(fi)
            
    # Get target attribute.
    target = askForTarget(data)

    # Perform ID3 algorithm on dataset.
    a = id3(target,data)
    a.buildTree()
    rules = a.tree.toStringAsLogic()
    print '%s\n\n--- End of output. ---' % (rules)
    
    # Write to rules file.    
    fh = open(RULES_FILE,'w')
    fh.write(rules)
    fh.close()
    print '--- Wrote to %s ---' % (RULES_FILE)
    
# If not imported.

if __name__ == '__main__':
    main()
