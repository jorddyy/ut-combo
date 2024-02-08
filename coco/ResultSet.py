#External imports
import os
import json
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats, special

#This code specific imports
from HelperFunctions import matrixPrettyPrint
from IntercorrelationMaps import IntercorrelationMaps


#-------------------------------------
def checkCorrmat( corrmat, dim ):
    """
    Helper function: Performs an integrity check of a correlation matrix
    Checks it is square, correct dimension, and elements make sense
    corrmat is a numpy matrix, dim is an integer
    """
    if( corrmat.shape[0] != corrmat.shape[1]  or corrmat.shape[0] != dim ):
        print('ERROR in checkCorrmat:  Matrix dimensions wrong')
        print('Matrix    dimension = ',corrmat.shape[0],' X ',corrmat.shape[1] )
        print('Parameter dimension = ',dim)
        return False
    
    passed = True
    for i in range(len(corrmat)):
        for j in range(len(corrmat)):
            elem  = round(corrmat[i,j],3)
            elemc = round(corrmat[j,i],3)
            if elem != elemc      :
                print('ERROR checkCorrmat: elem != elemconj for    [i,j]=[',i,',',j,'] elem=',elem,'  elemc=',elemc)
                passed = False
            if i==j and elem !=1. :
                print('ERROR checkCorrmat: diag elem not unity for [i,j]=[',i,',',j,'] elem=',elem,'  elemc=',elemc)
                passed = False
            if i!=j and elem  >1. :
                print('ERROR checkCorrmat: elem >1 for             [i,j]=[',i,',',j,'] elem=',elem,'  elemc=',elemc)
                passed = False

    return passed

#-------------------------------------
def readResultSetJSON( resset ):
    """
    MAIN USER FUNCTION:
    This parses a JSON ResultSet dictionary containing and transfers contents into a full ResultSet object
    There is purposely a lot of checking of the Json dictionary to try to help diagnose format errors.
    Inputs:
        resset: A JSON ResultSet dictionary
    Returns:
        A ResultSet
    """
    names = []
    values = []
    errors = []
    
    #Create minimal ResultSet
    try:
        label = resset['ResultSetLabel']
        desc = resset['Description']
        params = resset['Parameter']
    except:
        print('ERROR in readResultSetJSON: ResultSetLabel, Description or Parameter entries not found')
        print(resset)
        sys.exit()

    for param in params:
        try:
            names.append( param['Name'])
            values.append( param['Value'])
            errors.append( param['Error'])
        except:
            print('ERROR in readResultSetJSON: Reading ResultSet with label = '+label)
            print('Param found but cant decode Name, Value or Error properly')
            print(param)
            sys.exit()

    resultSet = ResultSet( label, desc, names, values, errors )

    #Add any statistical correlation matrix
    try:
        corrmat = np.matrix(resset['StatisticalCorrelationMatrix'])
    except:
        corrmat = np.matrix(np.diag(np.ones(len(values))))

    checkResult =  checkCorrmat( corrmat, len(values) )
    if( not checkResult ):
        print('ERROR in readResultSetJSON: Reading ResultSet with label = '+label)
        print('Problem with statistical correlation matrix')
        print(corrmat)
        sys.exit()

    resultSet.addStatCorrMat( corrmat )

    #Add any systematic errors
    try:
        systerrs = resset['SystematicErrors']
    except:
        systerrs=[]

    if( len(systerrs) > 0 ):
        for systerr in systerrs:
            try:
                name = systerr['Name']
                values = systerr['Values']
            except:
                print('ERROR in readResultSetJSON: Reading ResultSet with label = '+label)
                print('systerr found but cant decode Name + Values' )
                print(systerr)
                sys.exit()

            try:
                scorrmat = np.matrix(systerr['SystematicCorrelationMatrix'])
            except:
                scorrmat = np.matrix(np.diag(np.ones(len(values))))
            
            checkResult =  checkCorrmat( scorrmat, len(values) )
            if( not checkResult ):
                print('ERROR in readResultSetJSON: Reading ResultSet with label = '+label)
                print('Problem with a systematic correlation matrix for systErr = '+name)
                print(scorrmat)
                sys.exit()

            resultSet.addSystError( name, values, scorrmat )

    return resultSet


#--------------------------------------------------------------------
def readResultListJSON( filelist, icfilelist =[], inputpath='inputs/', outputpath='outputs/' ):
    """
    MAIN USER FUNCTION: Opens and reads file containing a complete JSON ResultList
    This contains all of the ResultSets which are to be averaged together.
    Inputs read from directory inputpath (default is subdirectory called inputs)
    Outputs put in directory outputpath (default is subdirectory called outputs)
       filelist: list of files to open from inputpath
    Returns a python ResultList
    """
    
    reslist = ResultList()
    
    print('\nreadResultListJSON: is reading the ResultSet files')
    
    #Check if outputs/ directory exists
    isthere = os.path.isdir(outputpath)
    if not isthere : os.mkdir(outputpath)
    
    #Open file to write a singles cmbined input file to
    outputfilename = outputpath+'CombinationInputOnefile.json'
    outputfile = open(outputfilename,'w')
    outputfile.write('{\n    "ResultSet": [ \n' )
    
    #Read all files and add result sets
    firstentry=True
    for file in filelist:
        print(file)
        json_data=open(inputpath+file).read()
        jdata = json.loads(json_data)
        #json.dump(jdata,outputfile,indent=2)
        ressets = jdata["ResultSet"]
        for resset in ressets:
            #Process one ResultSet
            resultSet = readResultSetJSON(resset)
            reslist.add(resultSet)
            
            #Add all ResultSets to one big output file
            if not firstentry : outputfile.write(',\n' )
            firstentry = False
            json.dump(resset,outputfile,indent=1)

    outputfile.write('\n]\n}')
    outputfile.close()
    print('\nreadResultListJSON:  Output written all files combined into:  '+outputfilename )

    #Read in and add all intercorrelation maps
    print('\nreadResultListJSON: is reading the Intercorrelartion configuration files')

    for icfile in icfilelist :
        print(icfile)
        json_data=open(inputpath+icfile).read()
        jdata = json.loads(json_data)
        rsLabels = jdata['ResultSetLabels']
        icdictList = jdata['IntercorrelationMaps']
        parameterEquivalenceLists = jdata['ParameterEquivalenceLists']
        reslist.addIntercorrelationMaps( rsLabels, icdictList, parameterEquivalenceLists )

    #reslist.setIntercorr( doInterCorr )

    #Do a sanity check
    if not reslist.sanityCheck(): sys.exit()

    return reslist

    
#--------------------------------------------------------------------
# Main class which represents a single ResultSet
#--------------------------------------------------------------------
class ResultSet:
    """
    Class to represent a single ResultSet
    This comprises a set of parameter estimates, stastistical and systematic errors and their correlations
    The user should normally not have to use this directly, but instead only use the ResultList class
    """
    
    def __init__(self, label, desc, names, values, errors ):
        """
        Constructor requires the minimal mandatory items for a ResultSet
         label : The unique name of the result set, e.g '2012-JPsiKK'
         names : List of the character names of all the parameters, e.g. ['gammas','deltaGammas',....]
         values: List of the corresponding values of each parameter
         error : List of the corresponding statistical errors of each parameter
        """
        if len(names) != len(values) or len(names) != len(errors) :
            print('ERROR in constructor of ResultSet: '+label)
            print('Parameter names/values/errors inconsistent length')
            x = input("Hit CR to continue  - DO NOT CONTINUE)")
            sys.exit()
        
        self.label = label
        self.description = desc
        self.paramNames = np.array(names)
        self.nparams = len(self.paramNames)
        self.paramValues = np.array(values)
        self.lastFitParamValues = np.array(values)
        self.paramErrors = np.array(errors)
        self.statCorrMat = np.matrix(np.diag(np.ones(self.nparams)))
        self.systErrorNames = []
        self.systErrorValues = []
        self.systCorrMats = []
        self.doStatCorr = True
        self.doSyst = True
        self.doSystCorr = True
 
 
    def addStatCorrMat(self, corrMat ):
        """
        Optional method to add a statistical correlation matrix
        A diagonal matrix is inserted in thre constructor as default.
        corrmat : must be an array like object with square dimension equal to numer of parameters
        """
        if not checkCorrmat( corrMat, self.length() ) :
            print('ERROR in addStatCorrMat: wrong size or shape: ')
            print('ResultSet name=',self.label, '  corrmat shape = ', corrMat.shape, '   self.nparams = ',self.length() )
            print(corrMat)
            x = input('Hit CR  - DO NOT CONTINUE)')
            sys.exit()
        self.statCorrMat = np.matrix(corrMat)

    
    def addSystError(self, name, errList, corrMat ):
        """
        Optional method to add a systematic error
        name   : character name of syst error, e.g. "FitBias"
        errlist: List of errors corresponding to each parameter (in same order)
        corrMat: correlation matrix
        """
        if len(errList) != self.length() :
            print('ERROR in addSystError: - syst error length !=  parameter length')
            print('Rset name=',self.label,'  systname=',name,'   len errlist=',len(errList),'   self.length=',self.length() )
            x = input('Hit CR  - DO NOT CONTINUE)')
            return
        if not checkCorrmat( corrMat, self.length() ) :
            print('ERROR in addSystError: wrong shape correlation matrix')
            print('Rset name=',self.label,'  systname=',name,' self.length=',self.length() )
            print(corrMat)
            x = input('Hit CR - DO NOT CONTINUE)')
            return
        
        self.systErrorNames.append(name)
        self.systErrorValues.append( np.array(errList) )
        self.systCorrMats.append( np.matrix(corrMat) )

        return
            

    def length(self):
        """Returns the number of parameters in the ResultSet"""
        return self.nparams
            
    def getLabel(self):
        """Returns the character name of the ResultSet e.g. '2012-JPsiKK'"""
        return self.label
    
    def getParameters(self):
        """Returns all of the character names of all the parameters as a 1-D numpy array"""
        return self.paramNames.copy()
    
    def getParameterIndex(self, param ):
        """Returns index of a parameter or -1 if it doesnt exist"""
        for i in range(self.length()) :
            if self.paramNames[i] == param : return i
        return -1

    def getValues(self):
        """Returns the values of all parameters as a 1-D numpy matrix"""
        return self.paramValues.copy()

    def getLastFitValues(self):
        """Returns the last fit values of all parameters as a 1-D numpy matrix"""
        return self.lastFitParamValues.copy()
    
    def getSystErrorNames(self):
        """Returns a list of the character names of all the systematic errors"""
        return np.array(self.systErrorNames)

    def hasSystErrorName(self, systType):
        """Determines if a particular systematic error name exists in thsi ResultSet"""
        for i in range(len(self.systErrorNames)):
            if self.systErrorNames[i] == systType: return True
        return False

    def setErrorTreatmentFlags( self, doStatCorr, doSyst, doSystCorr ):
        """Sets flags for how errors are treated"""
        self.doStatCorr = doStatCorr
        self.doSyst = doSyst
        self.doSystCorr = doSystCorr and doSyst
        if not doSyst and doSystCorr : print('\n WARNING in setting error flags: you tried doSyst=False with doSystCorr=True ')

    def getErrors(self):
        """
        Returns the quadrature sum of the statistical and systematic errors for all parameters
        Returned as a 1-D numpy array
        """
        statErrors = self.paramErrors
        if not self.doSyst: return statErrors   #BUG FIX PETE 220317
        systErrors = self.getSystErrors()
        totalErrors = np.sqrt( statErrors**2 + systErrors**2 )
        return totalErrors

    def getSystErrors(self):
        """
        Returns the quadrature sum of the systematic errors for all parameters
        Returned as a 1-D numpy array
        """
        if not self.doSyst : return np.array(np.zeros(self.length())) #BUG FIX PETE 220317
        errorssq = np.array(np.zeros(self.length()))
        for k in range(len(self.systErrorValues)):
            errorssq += self.systErrorValues[k]**2
        errors = np.sqrt(errorssq)
        return np.array(errors)


    def getSystTypeErrorSet(self, systType ):
        """
        Returns the set of systematic errors (for all parameters), corresponding to a given syst error type
        systType : character name of the syst error you want, e.g. 'FitBias'
        Returns a 1-D numpy array
        """
        if not self.doSyst: return np.array(np.zeros(self.length())) #BUG FIX PETE 220317
        for i in range(len(self.systErrorNames)):
             if self.systErrorNames[i] == systType:
                 return np.array(self.systErrorValues[i])
        return np.array(np.zeros(self.length()))


    def getResidualPulls(self):
        """
        Returns the residual pulls for all parameters w.r.t the last fit values
        Returned as a 1-D numpy array
        """
        values = self.getValues()
        errors = self.getErrors()
        lfvals = self.getLastFitValues()
        respulls = (values - lfvals)/errors
        return respulls

    def getCorrmatStat(self):
        """Returns the statistical correlation matrix as a 2-D numpy matrix"""
        if self.doStatCorr:
            return self.statCorrMat.copy()
        else:
            return np.matrix(np.diag(np.ones(self.nparams)))

    def getCorrmatSyst(self, systType):
        """ returns the correlation matrix corresponding to a specific systematic error"""
        if not self.hasSystErrorName(systType) : return np.matrix(np.diag(np.zeros(self.nparams)))
        if not self.doSyst :                     return np.matrix(np.diag(np.zeros(self.nparams)))
        if not self.doSystCorr :                 return np.matrix(np.diag(np.ones(self.nparams)))
        for i in range(len(self.systErrorNames)):
            if self.systErrorNames[i] == systType: return self.systCorrMats[i]

    def getCorrSystElem(self, systType, param1, param2):
        """Returns a the correlation of a systematic error between two specified parameters"""
        #See if the systematic error exists in this data set  -
        #If it exists then return the correlation between param1 and param2
        #If there is no such systematic then it returns zero. THIS IS A CHOICE. ONE COULD RETURN AN ERROR
        #If one or more of the parameters dont exist it returns an error value (>1). THIS IS A CHOICE TO RETURN AN ERROR (INVALID VALUE)
        if not self.doSyst: return 0.      #BUG FIX PETE 220317
        j = self.getParameterIndex( param1 )
        k = self.getParameterIndex( param2 )
        noParamCode = 10
        if j<0 or k<0 : return noParamCode
        if self.hasSystErrorName(systType) : return self.getCorrmatSyst( systType )[j,k]
        else : return 0.

    def getCovmatStat(self):
        """Returns the statistical covariance matrix as a 2-D numpy matrix"""
        mat = self.getCorrmatStat()
        for i in range(self.length()):
            for j in range(self.length()):
                mat[i,j]*=self.paramErrors[i]*self.paramErrors[j]
        return np.matrix(mat)
 
    def __getCovmatSyst__(self):
        """Private method as by itself this does not mean anything"""
        mat = np.zeros( (self.length(),self.length() ))
        if not self.doSyst: return mat
        for errtype in range(len(self.systErrorNames)):
            systType=self.systErrorNames[errtype]
            for i in range(self.length()):
                for j in range(self.length()):
                    mat[i,j]+=self.getSystTypeErrorSet(systType)[i]*self.getSystTypeErrorSet(systType)[j]*self.getCorrmatSyst(systType)[i,j]
        return np.matrix(mat)

    def getCovmat(self):
        """Returns the complete covariance matrix as a 2D numpy matrix"""
        matrix = self.getCovmatStat() + self.__getCovmatSyst__()
        return np.matrix(matrix)


    def getCorrmat( self ):
        """Returns the complete correlation matrix as a 2D numpy matrix"""
        errors = self.getErrors()
        mat = self.getCovmat()
        for i in range(self.length()):
            for j in range(self.length()):
                mat[i,j]/=(errors[i]*errors[j])
        return mat
    
    def getInvE(self):
        """Returns the inverse of the covariance matrix as a 2D numpy matrix"""
        matrix = self.getCovmat()
        return matrix.getI()
    
    def getDiff( self, fitDict ):
        """
        Returns the difference vector between the parameter values and a supplied set of test values
        This is aimed at those wishing to construct their own chis-squared
        testParameters: the supplied dictionary of { 'parameter names' : test values }
        The dictionary must contain at least all parameters in the ResultSet (it may have more)
        Returns a 1-D numpy matrix
        """
        #Check all the correct parameters have been passed in
        pars = self.getParameters()
        for p in pars:
            if p in fitDict :
                pass
                #Ok if this succeeded
            else:
                print('ERROR in getDiff: for ResultSet: ',self.label )
                print('Fit dictionary passed does not contain parameter ',p)
                x = input('Hit CR - DO NOT CONTINUE)')
                sys.exit()

        diff = []
        for i in range(self.length()):
            d = self.paramValues[i] - fitDict[self.paramNames[i]]
            self.lastFitParamValues[i] = fitDict[self.paramNames[i]]
            diff.append(d)
        return np.array(diff)
    
    
    def show(self):
        print('\n----------ResultSet------------')
        print('Label = ',self.label,'\n\nDescription:')
        for line in self.description: print(line)
        print('\n')
        paramDict = { }
        fstr = "{0:8.4f}"
        for i in range(self.length()):
            print("{:15s}".format(self.paramNames[i]), ':  \t', self.paramValues[i], ' +/- ', self.paramErrors[i], ' +/- ',fstr.format(self.getSystErrors()[i]), '     =      ', self.paramValues[i], ' +/- ', fstr.format(math.sqrt(self.paramErrors[i]**2 + self.getSystErrors()[i]**2)))
            paramDict.update( { self.paramNames[i] : [self.paramValues[i],math.sqrt(self.paramErrors[i]**2 + self.getSystErrors()[i]**2)] } )
        print('\nStatistical Correlation Matrix:')
        matrixPrettyPrint(self.statCorrMat,'f')
        return paramDict

    def printToFile(self):
        fstr = "{0:8.4f}"
        for i in range(self.length()):
            print("{:15s}".format(self.paramNames[i]), '\t', self.paramValues[i], ' \t', fstr.format(math.sqrt(self.paramErrors[i]**2 + self.getSystErrors()[i]**2)))

    
    def showAll(self, PCovMat=False):
        print('\n----------ResultSet------------')
        print('Label = ',self.label,'\n\nDescription:')
        for line in self.description: print(line)
        print('\n')
        fstr = "{0:8.4f}"
        for i in range(self.length()):
            print("{:15s}".format(self.paramNames[i]), ':  \t', self.paramValues[i], ' +/- ', self.paramErrors[i], ' +/- ',fstr.format(self.getSystErrors()[i]), '     =      ', self.paramValues[i], ' +/- ', fstr.format(math.sqrt(self.paramErrors[i]**2 + self.getSystErrors()[i]**2)))
        print('\nStatistical Correlation Matrix:')
        matrixPrettyPrint(self.statCorrMat,'f')
        if PCovMat : print('\nStatistical Covariance Matrix:')
        if PCovMat : matrixPrettyPrint(self.getCovmatStat())
        for i in range(len(self.systErrorNames)):
            print('\nSystematic Error:', self.systErrorNames[i])
            print(self.systErrorValues[i])
            matrixPrettyPrint(self.systCorrMats[i], 'f')
        if PCovMat : print('\nSystematic Covariance Matrix:')
        if PCovMat : matrixPrettyPrint(self.__getCovmatSyst__())
        print('\nCorrelation Matrix:')
        matrixPrettyPrint(self.getCorrmat(),'f')
        if PCovMat : print('\nCovariance Matrix:')
        if PCovMat : matrixPrettyPrint(self.getCovmat())


#--------------------------------------------------------------------
# Main class which represents a set of ResultSets
# This implements many combined operations.
#--------------------------------------------------------------------
class ResultList:
    """
    Class to contain a list of ResultSets to be combined into an average
    This will contain
        >=1 ResultSet objects (parameter values, stat+syst errors and all correlations)
    """
    
    def __init__(self):
        """Constructor creates empty ResultList"""
        self.resultList = []
        self.parameterTranslator = {}
        self.bigCovMat = np.matrix( (1, 1) ) #dummy
        self.bigCovMatI = np.matrix( (1, 1) ) #dummy
        self.bigCovMatIsDone = False  #This is for speed -  to avoid un-necessary recalculation of the covmat.
        self.doIntercorr = True
        self.verboseIntercorr = False
        self.intercorr = IntercorrelationMaps()
    
    def setErrorTreatmentFlags( self, *, doStatCorr, doSyst, doSystCorr, doIntercorr ):
        """Sets flags for how errors are treated"""
        for resset in self.resultList : resset.setErrorTreatmentFlags( doStatCorr, doSyst, doSystCorr )
        self.doIntercorr = doIntercorr
        self.bigCovMatIsDone = False
    
    def add(self,resultSet):
        """
        MAIN USER METHOD: To add an ResultSet into this ResultList
        There would not be much sense in using this code unless at least 2 ResultSets are added
        which are to be averaged.
        """
        self.resultList.append(resultSet)
    
    def addParameterTranslator( self, param, function ):
        """
        Populates a dictionary of parameter translatorT
        to be used when the average (re-fit) parameters do not match those in this ResultSet directly, but are linearly related
        """
        self.parameterTranslator.update( {param : function } )
    
    def addIntercorrelationMaps( self, rsLabels, icdict, parameterEqivalenceList ):
        self.intercorr.add(rsLabels, icdict, parameterEqivalenceList )
    
    def length(self):
        """Returns the total number of parameters in all the ResultSets"""
        len = 0
        for resultSet in self.resultList:
            len += resultSet.length()
        return len
    
    def getUniqueParameters(self):
        """
        Returns all of the unique parameter names contained within all ResultSets as a 1-d numpy array.
        For example, if 'gamma' appears in 2 ResultSets, only one 'gamma' is returned
        """
        parameters = np.array(())
        for resultSet in self.resultList:
            parameters = np.append( parameters, resultSet.getParameters() )
        return np.unique(parameters)

    def getParameters(self):
        """
        Returns the names of all parameters in the ResultList, taken from each ResultSet in order.
        Returned as a 1D numpy array
        User really doesnt need to use this
        """
        params = np.array(())
        for resultSet in self.resultList:
            params = np.append( params, resultSet.getParameters() )
        return params
 
    def getValues(self):
        """
        Returns the values of all parameters in the ResultList, taken from each ResultSet in order.
        Returned as a 1D numpy array
        """
        values = np.array(())
        for resultSet in self.resultList:
            values = np.append( values, resultSet.getValues() )
        return values

    def getLastFitValues(self):
        """
            Returns the values of all parameters in the ResultList, taken from each ResultSet in order.
            Returned as a 1D numpy array
            """
        lastFitValues = np.array(())
        for resultSet in self.resultList:
            lastFitValues = np.append( lastFitValues, resultSet.getLastFitValues() )
        return lastFitValues
    
    def getErrors(self):
        """
        Returns the total errors of all parameters in the ResultList, taken from each ResultSet in order.
        The user should not need this, as they really only need to covariance matrix.
        Returned as a 1D numpy array
        """
        errors = np.array(())
        for resultSet in self.resultList:
            errors = np.append( errors, resultSet.getErrors() )
        return errors

    def getResidualPulls(self):
        """
            Returns the residual pulls of all parameters in the ResultList with respect to the final fit values stored.
            These are taken from each ResultSet in order.
            Returned as a 1D numpy array
            """
        respulls = np.array(())
        for resultSet in self.resultList:
            respulls = np.append( respulls, resultSet.getResidualPulls() )
        return respulls
 
    def getStructure(self):
        """
         Returns the structure of thew ResultList: The label and dimension of each ResultSet
         These are taken from each ResultSet in order.
         """
        labels = []
        dimensions = []
        for resultSet in self.resultList:
            labels.append( resultSet.getLabel() )
            dimensions.append( resultSet.length() )
        return labels, dimensions

    def getUniqueSysErrors(self):
        """
        Returns all of the unique systematic error names contained within all ResultSets as a 1-d numpy array.
        For example, if 'FitBias' appears in 2 ResultSets, only one 'FitBias' is returned
        This is intended to help users if thwy need to work out which to correlate between ResultSets
        """
        errnames = np.array(())
        for resultSet in self.resultList:
            errnames = np.append( errnames, resultSet.getSystErrorNames() )
        return np.unique(errnames)


    def getCovMat( self ):
        """
        MAIN USER METHOD: Returns the full covariance matrix of the entire ResultList
        This is aimed at hose wishing to construct their own chisq
        """
        if self.bigCovMatIsDone : return self.bigCovMat
        
        bigCovMat = np.zeros( (self.length(), self.length()) )
 
        #Set up block diagonal covariance matrix
        offset = 0
        for resultSet in self.resultList:
            nparams = resultSet.length()
            covmat = resultSet.getCovmat()
            bigCovMat[offset:offset+nparams, offset:offset+nparams ] = covmat
            offset+= nparams

        # Add any inter ResultSet correlations
        if( self.doIntercorr ):
            for systType in self.getUniqueSysErrors():
                offset = 0
                for i in range(len(self.resultList)):
                    rs1 = self.resultList[i]
                    np1 = rs1.length()
                    offset2 = 0
                    if rs1.hasSystErrorName(systType) :
                        for j in range(i+1,len(self.resultList)):
                            rs2 = self.resultList[j]
                            np2 = rs2.length()
                            if rs2.hasSystErrorName(systType):
                                intercov = self.intercorr.getInterCovmat( systType, rs1, rs2, self.verboseIntercorr )
                                bigCovMat[offset+offset2+np1:offset+offset2+np1+np2, offset:offset+np1 ] += intercov.transpose()
                                bigCovMat[offset:offset+np1, offset+offset2+np1:offset+offset2+np1+np2 ] += intercov
                            offset2+=np2
                    offset+=np1
        
        self.bigCovMat = np.matrix(bigCovMat)
        self.bigCovMatI = self.bigCovMat.getI()
        self.bigCovMatIsDone = True
        return self.bigCovMat
        
    def getCovMatI(self):
        if self.bigCovMatIsDone :
            return self.bigCovMatI
        else:
            return self.getCovMat().getI()
    
    def getCorrMat( self ):
        """
        Returns the full correlation matrix of the entire ResultList
        """
        errors = self.getErrors()
        covmat = self.getCovMat()
        corrmat = np.copy(covmat)
        for i in range(self.length()):
            for j in range(self.length()):
                corrmat[i,j]/=errors[i]*errors[j]
        return corrmat

    def getDiff( self, fitDict ):
        """
        Returns the difference vector between the parameter values and a supplied set of test values
        This is aimed at those wishing to construct their own chi-squared
          testParameters: the supplied dictionary of { 'parameter names' : test values }
        The dictionary must contain at least all unique parameters in the ResultList
        Returns a 1-D numpy matrix
        """
        bigDiff = np.array(())
        for resultSet in self.resultList:
            bigDiff = np.append(bigDiff, resultSet.getDiff( fitDict ) )
        return np.array(bigDiff)

    def getChisq( self, fitDict ):
        """
        MAIN USER METHOD: Returns the chisq for all parameters in the ResultList tested against
        the supplied fitValues.
        [This is in correct format for MINUIT to use as its fcn]
          fitDict: A dictionary of supplied test parameter names and values
        """
        #Check all the correct parameters have been passed in, if not add from translators
        pars = self.getUniqueParameters()
        for p in pars:
            if p in fitDict :
                pass
            elif p in self.parameterTranslator :
                fun = self.parameterTranslator[p]
                v = fun(fitDict)
                # Add as new parameter the fit dictionary
                fitDict[p] = v
            else:
                print( 'ERROR in getChisq: fit dictionary does not contain parameter, and no translator for, ',p )
                x = input("Hit CR to continue  - DO NOT CONTINUE)")
                sys.exit()
   
        #Set up big difference vector
        bigDiff = np.matrix(self.getDiff(fitDict))
        #Get total covariance matrix  inverse
        bigCovMatI = self.getCovMatI( )
        #Calculate chisq
        chisq = bigDiff*bigCovMatI*bigDiff.transpose()
        return chisq[0,0]

            
    def getSimpleAverage( self ):
        """
        Obtains the simple average of the ResultList
        Two dictionaries are returned  {parametername:average} and {parametername:error}
        """
        #get the unique parameters to average
        upars = self.getUniqueParameters()
        vavg = []
        eavg = []
        #get all values and errors
        pars = self.getParameters()
        vals = self.getValues()
        errs = self.getErrors()
        #do average
        for upar in upars:
            nsum = 0.
            dsum = 0.
            for i in range(self.length()):
                if pars[i] == upar :
                    nsum += vals[i]/errs[i]**2
                    dsum += 1./errs[i]**2
            vavg.append( nsum/dsum )
            eavg.append( math.sqrt(1./dsum ))
        vresult = dict(list(zip(upars,vavg)))
        #vresult = sorted(vresult.keys())
        eresult = dict(list(zip(upars,eavg)))
        #eresult = sorted(eresult.keys())
        return vresult, eresult

    def getStartValues( self, params ):
        """
        Obtains start values and erros for the parameters in the list
        """
        svals = {}
        vals,errs =  self.getSimpleAverage()
        for param in params:
            try:
                v = vals[param]
                e = errs[param]
                svals.update({ param:v})
                svals.update({ 'error_'+param:e})
            except:
                svals.update({ param:0.})
                svals.update({ 'error_'+param:0.01})
        return svals

    def showAll(self, PrintCorrMat=False, PrintCovMat=False ):
        """Shows entire ResultList with all details"""
        print('\n--------------------------------------')
        print('ShowAll (all details)\n')
        for resultSet in self.resultList:
            resultSet.showAll()
        print('\n-----------Intercorrelations added---------\n')
        self.intercorr.show()
        self.bigCovMatIsDone = False
        self.verboseIntercorr = True
        corrmat = self.getCorrMat()
        covmat = self.getCovMat()
        if PrintCorrMat :
            print('\nTotal Correlation Matrix:')
            matrixPrettyPrint(corrmat,'f')
        if PrintCovMat:
            print('\nTotal Covariance Matrix:')
            matrixPrettyPrint(covmat)
        self.verboseIntercorr = False

    def show(self, PrintCovMat=False):
        """Shows parameters and overall correlation matrix"""
        print('\n--------------------------------------')
        print('Show (brief)\n')
        inputSets = []
        for resultSet in self.resultList:
            paramDict = resultSet.show()
            inputSets.append( paramDict )
        print('\n-----------Intercorrelations added---------\n')
        self.intercorr.show()
        self.bigCovMatIsDone = False
        self.verboseIntercorr = True
        corrmat = self.getCorrMat()
        if PrintCovMat:
            print('\nTotal Correlation Matrix:')
            matrixPrettyPrint(corrmat,'f')
        self.verboseIntercorr = False
        return inputSets

    def showSimpleAverage(self, paramorder =[]):
        """Shows the simple average of a ResultList"""
        vals, errs = self.getSimpleAverage()
        fstr = "{0:8.4f}"
        simpleAverage = {}
        print('Simple Average of ResultList:')
        if len(paramorder)==0 :
            paramorder = sorted(vals.keys())
        for key in paramorder:
            if key in vals :
                print('   {:15s}'.format(key), ':  \t',fstr.format(vals[key]),'  +-  ', fstr.format(errs[key]))
                simpleAverage.update( { key : [ vals[key], errs[key] ] } )
        return simpleAverage
            
    
    def outputSimpleAverage(self, outputFileName='SimpleAverageResultSet.json', title='SimpleAverageResultSet'):
        """ To write the simple average in the JSON format as a ResultSet """
        vals, errs = self.getSimpleAverage()
        fstr = "{0:8.4f}"

        fh = open( outputFileName, 'w' )
        fh.write( '{\n"ResultSet": [\n' )
        fh.write( '{\n\t"ResultSetLabel": "'+title+'",\n\t"Description": [ "'+title+'" ],' )
        fh.write( '\n\t"Parameter": [' )
    
        first = True
        for key in sorted(vals.keys()):
            if not first : fh.write( ',' )
            first = False
            fh.write( '\n\t{' )
            fh.write( '\n\t"Name": "'+key+'",')
            fh.write( '\n\t"Value": '+fstr.format(vals[key])+',')
            fh.write( '\n\t"Error": '+fstr.format(errs[key]))
            fh.write( '\n\t}' )
        fh.write( '\n\t]\n}\n]\n}')
        fh.close()
        print('Output file written:  JSON file containing simple average ResultSet: '+outputFileName)

    def sanityCheck(self):
        #Check the big correation matrix
        print('\nSanityCheck: Performing self sanity check of ResultList big correlation matrix')
        failed = False
        corrmat = self.getCorrMat()
        for i in range(len(corrmat)):
            for j in range(len(corrmat)):
                elem  = round(corrmat[i,j],3)
                elemc = round(corrmat[j,i],3)
                if elem != elemc      :
                    print('Sanity error: elem != elemconj for    [i,j]=[',i,',',j,'] elem=',elem,'  elemc=',elemc)
                    failed = True
                if i==j and elem !=1. :
                    print('Sanity error: diag elem not unity for [i,j]=[',i,',',j,'] elem=',elem,'  elemc=',elemc)
                    failed = True
                if i!=j and elem  >1. :
                    print('Sanity error: elem >1 for             [i,j]=[',i,',',j,'] elem=',elem,'  elemc=',elemc)
                    failed = True
        if failed :
            print('\nSanityCheck: FAILED - exiting ')
            x = input("Hit CR to continue - DO NOT CONTINUE)")
            sys.exit()
        print('\nSanityCheck: PASSED')
        return True
