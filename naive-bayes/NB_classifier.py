#!/bin/python

# NB_classifier.py
# -------------------------
# Winter 2013; Alex Safatli
# -------------------------
# A program that carries out
# the NB algorithm on a
# given data file(s).
#
# Usage: python NB_classifier.py

# Imports

import os, sys

# Constants

RESULT_FILE = 'Result.txt'

# Naive Bayesian Class(es)

class naivebayesian:
    def __init__(self,target,train,test):
        self.data = train
        self.data_target = train[target]
        self.test = test
        self.target = target
        self.target_classes = set(train[target])        
        self.count = {}
        self.prior = {}
        self.accuracy = (0,0)
        self.learned = False
    def calcPriorForAttr(self,attr):
        targd = self.data_target
        attrd = self.data[attr]
        classes = set(attrd)
        self.count[attr] = {}
        self.prior[attr] = {}
        # Set up dictionary.
        for cla in classes:
            # For class in attribute.
            self.count[attr][cla] = {}
            self.prior[attr][cla] = {}
            for tla in self.target_classes:
                # For class in target.
                self.count[attr][cla][tla] = 0
        # Do counts for every item in
        # list for attribute.
        for it in xrange(len(attrd)):
            self.count[attr][attrd[it]][targd[it]] += 1
        # Calculate prior probabilities
        # for all classes in attr.
        for cla in classes:
            # For class in attribute.
            for tla in self.target_classes:
                # For class in target.
                clacnt = self.count[attr][cla][tla]
                tarcnt = self.count[self.target][tla]
                if (tarcnt == 0):
                    prior = 0
                else:
                    prior = clacnt/float(tarcnt)
                self.prior[attr][cla][tla] = prior
    def calcPostsForTrans(self,dic):
        # Given dictionary of header:value pairs
        # for a transaction, dic, calculate post-
        # erior probabilities.
        postprobs = {}
        for tla in self.target_classes:
            postprobs[tla] = self.prior[self.target][tla]
            if postprobs[tla] == 0:
                # use m-estimator for handling missing values
                # in training data with m = 2
                postprobs[tla] = (2*(1/float(len(\
                    self.prior[self.target].keys())+1))/float(2))
            for attr in dic:
                if dic[attr] in self.prior[attr]:
                    postprobs[tla] *= self.prior[attr][dic[attr]][tla]
                else:
                    # use m-estimator
                    mest = (2*(1/float(len(\
                        self.prior[attr].keys())+1))/float(2))
                    postprobs[tla] *= mest
        return postprobs
    def learn(self):
        # Calculate prior probabilities
        # on the basis of training data.
        targetd = self.data_target
        self.count[self.target] = {}
        self.prior[self.target] = {}
        # Do counts for the target.
        for cla in self.target_classes:
            clacnt = len([x for x in targetd if x==cla])
            self.count[self.target][cla] = clacnt
            self.prior[self.target][cla] = clacnt/float(len(targetd))
        # Do counts for all other items.
        for attr in self.data:
            if attr != self.target:
                # If not target.
                self.calcPriorForAttr(attr)
        self.learned = True
        del self.count # clear count data; no longer needed.
    def predict(self):
        # Predict class given learned data.
        if not self.learned:
            raise SystemError("Cannot predict yet. Need to train.")
        # Get number of transactions.
        numtrans = len(self.test[self.test.keys()[0]])
        arrname = self.target[0:5]+'_classified'
        self.test[arrname] = [None for x in xrange(numtrans)]
        # For each transaction, predict.
        numsame = 0
        for tr in xrange(numtrans):
            trans = {}
            for attr in self.test:
                if attr != self.target and attr != arrname:
                    trans[attr] = self.test[attr][tr]
            posts = self.calcPostsForTrans(trans) # get posterior probs
            argmax = max(posts,key=lambda x:posts[x]) # argmax
            self.test[arrname][tr] = argmax
            if self.target in self.test and \
               self.test[self.target][tr]==argmax:
                numsame += 1
        # Report classification accuracy.
        self.accuracy = (numsame,numtrans)

# Helper Class(es) and Function(s)

def loadFile(fi):
    # Load a file and generate data.    

    # Get data from file.
    fh = open(fi,'r')
    lines = fh.read().split('\n')
    fh.close()
    values = []
    data = {}
    
    if len(lines) == 0:
        return data
    
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

def writeFile(fo,data,acc=None):
    out = ""
    # Get number of transactions.
    numtrans = len(data[data.keys()[0]])
    # Write headers.
    i = 0
    for attr in data:
        if (i == 0):
            out += "%s" % (attr)
        else:
            out += " %s" % (attr)
        i += 1
    out += "\n"
    # Write values.
    for tr in xrange(numtrans):
        i = 0
        for attr in data:
            if (i == 0):
                out += "%s" % (data[attr][tr])
            else:
                out += " %s" % (data[attr][tr])
            i += 1
        out += "\n"
    if acc:
        out += "Accuracy: %d/%d\n" % (acc[0],acc[1])
    fh = open(fo,"w")
    fh.write(out)
    fh.close()
    return out

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
    
    # Get all keys.
    keys = dataset.keys()
    # Print as a list to the user.
    print '\nPlease choose an attribute (by number):'
    for key in keys:
        print '\t%d. %s' % (keys.index(key)+1,key)
    # Get list index + 1.
    val = ask(len(keys))
    return keys[val-1]

# Main Function

def main():
    
    # Get user input for training and testing data files.
    trfi = raw_input('Specify the path for the (training) data file: ').replace('\ ',' ')
    if not os.path.isfile(trfi):
        print '<%s> is not a file. Check the provided path.' % (trfi)
        exit()    
    tefi = raw_input('Specify the path for the (testing) data file: ').replace('\ ',' ')
    if not os.path.isfile(tefi):
        print '<%s> is not a file. Check the provided path.' % (tefi)
        exit()    
    
    # Get datasets.
    trdata = loadFile(trfi)
    if (len(trdata.keys()) == 0):
        # Empty set.
        print '<%s> was parsed to an empty set. Cannot train on.' % (trfi)
        exit()    
    tedata = loadFile(tefi)
    if (len(tedata.keys()) == 0):
        # Empty set.
        print '<%s> was parsed to an empty set. Nothing to test.' % (tefi)
        exit()
        
    # Check if datasets have common attributes.
    trset = set(trdata.keys())
    teset = set(tedata.keys())
    if not (len(trset.difference(teset))==0):
        print '<%s> did not possess common attributes with <%s>.' % (tefi,trfi)
        print '<%s> contained attributes: %s' % (trfi,list(trset))
        print '<%s> contained attributes: %s' % (tefi,list(teset))
        exit()
        
    # Get target attribute.
    target = askForTarget(trdata)

    # Perform NB algorithm on dataset.
    a = naivebayesian(target,trdata,tedata)
    a.learn() # Calculate prior probabilities based on training data.
    a.predict() # Predict target class for given test data transactions.
    writeFile(RESULT_FILE,a.test,a.accuracy)
    print '\nAccuracy found: %d/%d.\nThe result is in the file %s.' \
          % (a.accuracy[0],a.accuracy[1],RESULT_FILE)
    
# If not imported.

if __name__ == '__main__':
    main()
