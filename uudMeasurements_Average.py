# libraries
import json
import sys, os
sys.path.append('coco')
sys.path.append('inputs')
from ResultSet import readResultListJSON
from HelperFunctions import EvaluateForMinuit, printChi2nDoF, showIminuitResult, showIminuitResultCorrmat, outputMinuitResult
from iminuit import Minuit

def combiner(mode, inputFolder, outputFolder):
    print(" ----------------- Combining ----------------- ")
    print( f"Mode: {mode}" )
    print( f"Input folder: {inputFolder}" )
    print( f"Output folder: {outputFolder}" )
    print( "--------------------------------------------- " )

    inputFile = f"ResultList_{mode}.json"

    reslist = readResultListJSON([inputFile], inputpath = inputFolder, outputpath = outputFolder)

    pars = reslist.getUniqueParameters()

    #Show all systematic errors and full correlation matrix
    reslist.showAll(PrintCorrMat=True,PrintCovMat=True)

    print(f'\n-------- Doing the combination {mode} ---------')

    measLabels, _ = reslist.getStructure()
    if len(measLabels) == 1:
        print('Only one measurement, nothing to combine.')
        return 0

    reslist.setErrorTreatmentFlags( doStatCorr=True, doSyst=True, doSystCorr=True, doIntercorr=False )

    eval = EvaluateForMinuit( reslist, pars )

    svals = reslist.getStartValues( pars )
    start =[ svals[p] for p in pars ]

    m = Minuit(eval.fcn, start, name=pars)
    m.errordef = 1.0
    m.print_level = 0
    m.tol = m.tol*0.0001

    m.migrad()
    m.minos()
    showIminuitResult(m)
    showIminuitResultCorrmat(m)

    chisqndof = printChi2nDoF(reslist, m)
    
    # add path and file
    outputpath = os.path.join(outputFolder, f"AverageResultSet_{mode}.json")
    outputMinuitResult( m, outputFileName=outputpath, title='Comb', minos=True )


if __name__ == "__main__":
    # Define which modes to combine
    modeList = [
        "pi+pi-",
        "rho+rho-",
        "rho0rho0",
        "a1+-pi-+",
    ]

    for mode in modeList:
        combiner(mode=mode,
                 inputFolder="inputs/uudMeasurements/",
                 outputFolder="outputs/uudMeasurements/")