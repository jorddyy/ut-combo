import sys
import numpy as np
import json
import statistics
import math

DIAG='DIAG'
FULL='FULL'

class IntercorrelationMaps:
    """
    Class to contain all of the logic to implement intercorrelations between different ResultSet s
    In simple circumstances the users does not need to alter this
    The users only needs to supply the JS|ON  configuration files to declare
        Systematics to be intercorrelated between given ResultSets
        Parameter quivalences (parameters with different names in different ResultSets to be considered as the same for correlation)
    """

    def __init__(self) :
        self.rsLabelsSet = []
        self.intercorrelationMapsSet = []
        self.parameterEquivelenceListsSet = []
        self.mutable_icdict={}
        self.mutable_parameterEquivalenceLists = []

    
    def add( self, rsLabelPair, intercorrelationMapsList, parameterEquivelenceLists ):
        """
        Adds the contents of a JSON configuration file
        """
        self.rsLabelsSet.append(rsLabelPair)
        icdict = {}
        for item in intercorrelationMapsList : icdict.update(item)
        self.intercorrelationMapsSet.append(icdict)
        for item in parameterEquivelenceLists : self.parameterEquivelenceListsSet.append(item)
        
    def length(self):
        return len(self.rsLabelsSet)



    def getInterCovmat( self, systType, rs1, rs2, Verbose=False  ):
        """
        This is the principle entry point - called from ResultSet.py
        It returns the off diagonal intercorrelated covariance matrix between two ResultSets corresponding to a given systematic type
        It is driven by the JSON configuraiton files read and stores at initialisation time (inserted by add method)
        """
        rsl1=rs1.getLabel()
        rsl2=rs2.getLabel()
        len1 = rs1.length()
        len2 = rs2.length()
        
        if not (rs1.hasSystErrorName(systType) and rs2.hasSystErrorName(systType)) : return self.nullMap( len1, len2 )
        
        #Called to set up the dictionary (icdict) of intercorrelations for these two ResultSets +  the parameter eqwuivalences.
        #Reults put in class variables so that they are available ot other methods
        self.__setupPair( rsl1, rsl2 )
        icdict =  self.mutable_icdict
        if len(icdict) == 0 : return self.nullMap( len1, len2  )

        if systType in icdict.keys():
            attributes = icdict[systType]
        else:
            if Verbose: print('IntercorrelationMaps: rs1='+rsl1+'  rs2='+rsl2+' : no correlation map for : '+systType)
            return self.nullMap( len1, len2 )

        #Create 100% intercorrelated covariance matrix as basis
        elist1 = np.matrix(rs1.getSystTypeErrorSet( systType ))
        elist2 = np.matrix(rs2.getSystTypeErrorSet( systType ))
        intercov = elist1.transpose()*elist2

        #Get correlation factors
        if attributes['ictype'] == DIAG :
            if Verbose: print('IntercorrelationMaps: rs1='+rsl1+'  rs2='+rsl2+' :  applied diagonal intercorr for: '+systType)
            intercorrmat = self.diagonalMap( rs1, rs2, attributes )
            return np.multiply(intercov,intercorrmat)
        elif attributes['ictype'] == FULL :
            if Verbose: print('IntercorrelationMaps: rs1='+rsl1+'  rs2='+rsl2+' : applied full intercorr for: '+systType)
            intercorrmat = self.fullMap( systType, rs1, rs2, attributes  )
            return np.multiply(intercov,intercorrmat)
        else:
            print('EXITING: IntercorrelationMaps: for systType=',systType,'  ictype=',ictype,' not known ')
            sys.exit()


    def __setupPair( self, rsl1, rsl2 ):
        """ Internal method. NUST be called for each new pair of ResultSets input to the main entry point."""
        icdict = {}
        pel = []
        for i in range(self.length()) :
            labelPair = self.rsLabelsSet[i]
            if( (labelPair[0]=='All' and labelPair[1]=='All') or ( rsl1 in labelPair and rsl2 in labelPair ) ):
                icdict.update(self.intercorrelationMapsSet[i])
                pel.append(self.parameterEquivelenceListsSet[i])
        self.mutable_icdict =  icdict
        self.mutable_parameterEquivalenceLists  = pel
    

    #These are all the possible mappings
    
    def nullMap( self, len1, len2 ):
        """ Intercorrelation map: sets no correlations at all"""
        map = np.zeros((len1,len2))
        return np.matrix(map)


    def diagonalMap( self, rs1, rs2, attributes ):
        """
        Intercorrelation map: sets diagonal correlations -
        i.e only correlations between same (equivalent) parameter in each ResultSet
        """
        params1 = rs1.getParameters()
        params2 = rs2.getParameters()
        map = np.zeros((len(params1),len(params2)))
        if 'scale' in attributes : factor =  attributes['scale']
        else : factor = 1.
        
        for i in range(len(params1)):
            for j in range(len(params2)):
                if( self.isEquivalent( params1[i] ,params2[j] ) ): map[i,j] = factor
        return np.matrix(map)

    def fullMap( self, systType, rs1, rs2, attributes ):
        """
        Intercorrelation map: sets full correlations
        I.e. sets correlations between every parameter in each ResultSet for this systematic error
        Searches for the correlation in question within each pair of parameters in each ResulttSet
        and sets the lower of the two as the intercorrelation between ResultSets
        """
        params1 = rs1.getParameters()
        params2 = rs2.getParameters()
        map = np.zeros((len(params1),len(params2)))
        if 'scale' in attributes : factor =  attributes['scale']
        else : factor = 1.
        if 'strategy' in attributes: strat = attributes['strategy']
        else:  strat = 'SQRT'
        
        for i in range(len(params1)):
            for j in range(len(params2)):
                c1 = self.getCorrSystElem( rs1, systType, params1[i], params2[j])
                c2 = self.getCorrSystElem( rs2, systType, params1[i], params2[j])
                if( abs(c1)<=1.0 and abs(c2)<=1.0):
                    #Both ResultSets contain the systematic and have a valid correlation between the elements
                    #if c1 or c2 wasnt found it would have been set to > 1
                    if strat == 'AVG' : rho = statistics.mean( [c1,c2])
                    elif strat == 'MIN' and ( abs(c1) < abs(c2) ) : rho = c1
                    elif strat == 'MIN' and ( abs(c1) >= abs(c2) ) : rho = c2
                    elif strat == 'MAX' and ( abs(c1) < abs(c2) ) : rho = c2
                    elif strat == 'MAX' and ( abs(c1) >= abs(c2) ) : rho = c1
                    elif strat == 'SQRT' :
                        if np.sign(c1) == np.sign(c2) :
                            rho = math.sqrt(c1*c2) * np.sign(c1)
                        else:
                            rho = statistics.mean([c1,c2])
                    else:
                        print("WARNING -  in intercorrelations fullMap : illegal strategy: "+strat)
                        sys.exit()
                    map[i,j] = rho*factor
                else:
                    #no valid correlation, i.e at least one of c1 and c2 were > 1
                    map[i,j] = 0.
        return np.matrix(map)

                    
    def isEquivalent( self, p1, p2 ):
        """
        This tests is two parameters (from different ResultSets) are the same.
        The direct comparison is easy. The complicaiton is when parameters with different
        names in each ResultSet are equivalent for correlation purposes
        Thus it needs to search the equivalence lists.
        """
        if p1 == p2 : return True
        #If not then search to see if both are in the same equivalance list
        for el in self.mutable_parameterEquivalenceLists:
            if (p1 in el) and (p2 in el) : return True
        return False

    
    def getCorrSystElem( self, rs, systType, p1, p2 ):
        """
        This gets the correlation between two parameters in a ResultSet
        The complicaiton is when parameters with different names in each ResultSet are equivalent.
        Thus it needs to search the equivalence lists.
        """
        #Does this systeElem exist ? if not return zero
        if not rs.hasSystErrorName( systType ) :  return 0.
        #If it does then if the parameters both exist return the correlation
        corr = rs.getCorrSystElem( systType, p1, p2)
        if abs(corr) <= 1.0 : return corr
        #If corr >  1 it means one or more paramerters did not exist, so try the equivalentce lists
        p1list=[p1]
        p2list=[p2]
        for el in self.mutable_parameterEquivalenceLists:
            if (p1 in el)  : p1list = list(el)
            if (p2 in el)  : p2list = list(el)
        #Do all combinations in equivalance list and return as soon as first valid correlation is found
        for pa in p1list :
            for pb in p2list:
                corr = rs.getCorrSystElem( systType, pa, pb)
                if abs(corr) <= 1.0 :  return corr
        return 10


    def show(self):
        print('\nIntercorrleation Maps read in:')
        for i in range(self.length()) :
            labelPair = self.rsLabelsSet[i]
            print('\nFor rs1 = ',labelPair[0],'  rs2 = ',labelPair[1])
            print('\nThe set of corrrelation maps is: ')
            print(self.intercorrelationMapsSet[i])
            print('\nThe set of parameter equivalences is: ')
            print(self.parameterEquivelenceListsSet[i])
            print('\n')
 
