import os
import json
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats, special



#-----------------------------------------------
#This tiny class is just to interface to Minuit
#It provides an "fcn" function needed by Minuit
class EvaluateForMinuit:
    def __init__(self, reslistToUse, parametersToFit ) :
        #fitpars are the list of names of the target average parameters
        #It must also be given to the Minuit constructor in the same order
        self.fitpars = parametersToFit
        self.reslist = reslistToUse
    def fcn( self, vals ):
        #vals are the current iteration of the target average parameters (in same order as fitpars)
        fitDict = dict(list(zip(self.fitpars,vals)))
        return self.reslist.getChisq( fitDict )

def getVaryParams(m):
    varyParams = []
    for par in m.params:
        if par.is_fixed == False:
            varyParams.append(par.name)

    return varyParams

def getCorrMatWithoutFixed(m):
    corrmat = np.array(m.covariance.correlation())
    number = 0
    for par in m.params:
        if par.is_fixed == True:
            corrmat = np.delete(corrmat, number, axis=0)
            corrmat = np.delete(corrmat, number, axis=1)
        else:
            number+=1
    return corrmat

#-----------------------
#Make residuals plots
def makeResidualPlot( chisqndof, reslist, minfit, output="outputs/PullPlot.pdf", filter=[] ):
    #Get parameters and residuals, filtered if requested

    #This step is needed to ensure the reslist has the last best-fit values set everywhere
    #This is the easiest way to do it as it forces all the translators to be called.
    pars = getVaryParams(minfit)
    vals = minfit.values
    fitDict = dict(list(zip(pars,vals)))
    reslist.getChisq(fitDict)

    #Now the reslist has been properly set, continue.
    params = reslist.getParameters()
    respulls = reslist.getResidualPulls()
    if len(filter)==0 : filter = params.copy()
    r = []
    x = []
    for i in range(len(params)):
        if params[i] in filter:
            x.append(params[i])
            r.append(respulls[i])

    #Get structure of resultlist, i.e the labels and dimensions of each resultset
    labels, dims = reslist.getStructure()
    #Make a cumulative bounds list
    for i in range(1,len(dims)) : dims[i]+=dims[i-1]
    dims.insert(0,0)

    #find largest pull
    maxpull = abs(max(r))
    maxpull+=0.7

    #Set up the plot
    lcol = [ 'b', 'r', 'g', 'c', 'm', 'y', 'k', 'b', 'r', 'g', 'c', 'm', 'y', 'k', 'b', 'r', 'g', 'c', 'm', 'y', 'k']
    mkr = [ 'bo', 'ro', 'go', 'co', 'mo', 'yo', 'ko', 'bv', 'rv', 'gv', 'cv', 'mv', 'yv', 'kv', 'bs', 'rs', 'gs', 'cs', 'ms', 'ys', 'ks']
    fig, axs = plt.subplots(1,1, gridspec_kw={'bottom': 0.3})
    axs.set_ylim(-maxpull,maxpull)
    plt.xticks(rotation=60, ha='right')
    topline =' Chisq/ndof ='+str(chisqndof)
    axs.set_xlabel(topline)
    axs.xaxis.set_label_position('top')

    # Add a horizontal line at y=0
    axs.axhline(0, color='black', lw=0.5)


    #Add each ResultSet with a different colour
    for i in range(len(dims)-1):
        axs.stem(x[dims[i]:dims[i+1]], r[dims[i]:dims[i+1]], lcol[i], markerfmt=mkr[i], label=labels[i], linefmt=lcol[i], basefmt=' ')
    axs.legend( prop={'size': 6} )
    plt.savefig(output)
    plt.show()

#-------------
#Helper function
def printLatexTable(m, latexLabels, outputFileName='CombinationCorrmatTable.tex' ):

    valDict =  m.values
    errDict = m.errors
    paramnames =  getVaryParams(m)
    corrmat = getCorrMatWithoutFixed(m)

    fh = open(outputFileName, 'w')
    for i in range(len(paramnames)):
        line=""
        line+=latexLabels[paramnames[i]]+' '
        for j in range(len(paramnames)):
            if j>=i :
                sign = ""+(corrmat[i][j]<0)*"$-$"
                line+=' & '+ sign +str(abs(round(corrmat[i][j], 2)))
            else : line+=' & '
        line+='\\\\ \n'
        fh.write(line)
    fh.close()
    print('Output file written:  Latex Table containing core of correlation matrix: '+outputFileName)


#-------------
#Helper function
def showIminuitResult( m ):
    #averageList = {}
    valDict = m.values
    errDict = m.errors
    paramNames =  getVaryParams(m)

    print('\nIMinuit average ')
    fstr = "{0:8.4f}"
    for pn in paramNames :
        print('   {:15s}'.format(pn), ':  \t', fstr.format(valDict[pn]), ' +/- ', fstr.format(errDict[pn]))
    #averageList.update( { pn : [ valDict[pn], errDict[pn] ] } )
    print('\n')
    #return averageList

#-------------
#Helper function
def showIminuitMinosResult( m ):
    valDict = m.values
    errDict = m.merrors
    paramNames =  getVaryParams(m)
    print('\nIMinuit MINOS average ')
    fstr = "{0:8.4f}"
    for pn in paramNames :
        print('   {:15s}'.format(pn), ':  \t', fstr.format(valDict[pn]), ' +'+fstr.format(errDict[pn].upper), fstr.format(errDict[pn].lower))
    print('\n')


#-------------
#Helper function
def showIminuitResultCorrmat( m ):
    print('\n Final Corrmat: ')
    corrmat = np.matrix(m.covariance.correlation())
    matrixPrettyPrint( corrmat, 'f' )


#-----------------------------------
#Helper function
def outputMinuitResult( m, outputFileName='CombinationOutputResultSet.json', title='CombinationOutputResultSet', minos=False):
    """ To write the minuit output in the JSON format as a ResultSet """

    valDict =  m.values
    if(minos):
        errDict = m.merrors
    else:
        errDict = m.errors
    cm = list(getCorrMatWithoutFixed(m))
    paramNames =  getVaryParams(m)

    fstr = "{0:10.6f}"

    fh = open(outputFileName, 'w' )
    fh.write( '{\n"ResultSet": [\n' )
    fh.write( '{\n\t"ResultSetLabel": "'+title+'",\n\t"Description": [ "'+title+'" ],' )
    fh.write( '\n\t"Parameter": [' )

    first = True
    for param in paramNames :

        if not first : fh.write( ',' )
        first = False
        fh.write( '\n\t{' )
        fh.write( '\n\t"Name": "'+param+'",')
        fh.write( '\n\t"Value": '+fstr.format(valDict[param])+',')
        if(minos):
            fh.write( '\n\t"Error": '+fstr.format(max(errDict[param].upper,errDict[param].lower)))
        else:
            fh.write( '\n\t"Error": '+fstr.format(errDict[param]))
        fh.write( '\n\t}' )
    fh.write( '\n\t],')
    fh.write( '\n\t"StatisticalCorrelationMatrix": [')

    fstr = "{0:3.2f}"
    first = True
    for i in range(len(cm)):
        if not first : fh.write( ',' )
        first = False
        line = "  "
        first2 = True
        for j in range(len(cm)):
            if not first2 : line+=','
            first2 = False
            if cm[i][j] <0: line += fstr.format(cm[i][j])
            else: line += " " + fstr.format(cm[i][j])
        fh.write( '\n\t[ '+line+' ]' )
    fh.write( '\n\t]\n}\n]\n}')
    fh.close()
    print('Output file written:  JSON file containing final combined ResultSet: '+outputFileName)



#---------------------------------
#Helper function
def dictPrettyPrint( dict, form='e' ):
    """To print a python dictionary in pretty format """
    if form == 'e': fstr = "{0:8.4e}"
    if form == 'f': fstr = "{0:8.4f}"
    print('Dictionary contents in alphabetical order')
    for key in sorted(dict.keys()):
        print('  ',key,' : ',fstr.format(dict[key]))

#-----------------------------------
#Helper function
def matrixPrettyPrint( inmat, form='e' ):
    """
    To print out correlation matrices in a pretty format
    form='e' uses scientific notaiton, form='f' uses fixed decimal
    """
    mat = np.copy(inmat)
    list = mat.tolist()
    if form == 'e': fstr = "{0:7.1e}"
    if form == 'f': fstr = "{0:4.2f}"
    for i in range(mat.shape[0]):
        line = "  "
        for j in range(len(list)):
            if( form =='e' ):  #line += fstr.format(list[i][j]) + "    "
                if( j < i ): line += "\t\t"
                elif list[i][j] <0: line += fstr.format(list[i][j]) + "\t"
                else: line += " " + fstr.format(list[i][j]) + "\t"

            elif( form =='f' ):
                if( j < i ): line += "\t"
                elif list[i][j] <0: line += fstr.format(list[i][j]) + "\t"
                else: line += " " + fstr.format(list[i][j]) + "\t"
        print(line)



#--------------------------------------------------------------------
def printChi2nDoF(reslist, m):
    chi2 = m.fval

    nFitParams = len( getVaryParams(m))

    nInputs = len(reslist.getParameters())

    nDoF = float(nInputs-nFitParams)

    p =  (1 - stats.chi2.cdf(chi2,nDoF))

    print("nInputs =", nInputs, "nFitParams=", nFitParams)
    print("chi2 =", np.round(chi2,3), "nDoF =", np.round(nDoF,3), "chi2/nDoF =", np.round(chi2/nDoF,3), "p-value =", np.round(p,6))

    return np.round(chi2/nDoF,3)

#-----------------------
#Make 1D profile plots
def make1dProfilePlot(xval, yval, bestfit=None, error=None, ax=None, lopts={}, fopts={}):

    ax.fill_between( xval, 0, yval, **fopts )
    ax.plot( xval, yval, **lopts )
    return None
