#!/bin/sh 
WORKDIR=WORKDIR_TEMPLATE

source /work/stack/loadLSST.bash 
setup pfs_pipe2d VERSION_TEMPLATE
#setup fluxmodeldata
setup fluxmodeldata 20230608-small
setup display_matplotlib 
setup -jr WORKDIR_TEMPLATE/src/drp_qa 

cd $WORKDIR
sleep 60

####################
## DRP processing ##
####################

echo "Start processing..."
date

## run pfs_pipe2d (up to pfsMerged) ##
pipetask --long-log --log-level pfs=INFO run --register-dataset-types -b /work/datastore --instrument lsst.obs.pfs.PrimeFocusSpectrograph -i INPUT_TEMPLATE -o OUTPUT_TEMPLATE/VISITS_TEMPLATE -p 'WORKDIR_TEMPLATE/configs/calibrateExposureObsproc.yaml#isr,cosmicray,reduceExposure,mergeArms' -d "visit = VISITS_TEMPLATE" --fail-fast -j 12 -c parameters:h4_doCR=False --rebase

## fluxCalQa (with pfsMerged) ##
cd /lfs/work/pfs/s25a/figures/drpQa/fluxCalQa
/lfs/work/pfs/s25a/src/drp_qa/python/pfs/drp/qa/fluxCal/fluxCalQA.py OUTPUT_TEMPLATE/VISITS_TEMPLATE VISITS_TEMPLATE --skipFluxCal --saveData
mv fluxCalQA_vVISITS_TEMPLATE.csv fluxCalQA_vVISITS_TEMPLATE.tmp.csv
mv fluxCalQA_vVISITS_TEMPLATE.png fluxCalQA_vVISITS_TEMPLATE.tmp.png

## run pfs_pipe2d (up to pfsReference) ##
pipetask --long-log --log-level pfs=INFO run --register-dataset-types -b /work/datastore --instrument lsst.obs.pfs.PrimeFocusSpectrograph -i OUTPUT_TEMPLATE/VISITS_TEMPLATE -o OUTPUT_TEMPLATE/VISITS_TEMPLATE -p 'WORKDIR_TEMPLATE/configs/calibrateExposureObsproc.yaml#fitFluxReference' -d "visit = VISITS_TEMPLATE" --fail-fast -j 12 --extend-run 

##############
## calc EET ##
##############

echo "Calculate EET..."
date

conda deactivate
source /lfs/work/pfs/.venv/bin/activate
/lfs/work/pfs/.venv/bin/python WORKDIR_TEMPLATE/scripts/calc_eet.py --workDir WORKDIR_TEMPLATE --config CONFIG_TEMPLATE --skipAg --visits VISITS_TEMPLATE --updateDB 
deactivate

## run pfs_pipe2d (up to pfsCalibrated) ##

source /work/stack/loadLSST.bash 
setup pfs_pipe2d VERSION_TEMPLATE
#setup fluxmodeldata
setup fluxmodeldata 20230608-small
setup display_matplotlib 
setup -jr WORKDIR_TEMPLATE/src/drp_qa 

pipetask --long-log --log-level pfs=INFO run --register-dataset-types -b /work/datastore --instrument lsst.obs.pfs.PrimeFocusSpectrograph -i OUTPUT_TEMPLATE/VISITS_TEMPLATE -o OUTPUT_TEMPLATE/VISITS_TEMPLATE -p 'WORKDIR_TEMPLATE/configs/calibrateExposureObsproc.yaml#fitFluxCal' -d "visit = VISITS_TEMPLATE" --fail-fast -j 12 -c fitFluxCal:fitFocalPlane.polyOrder=3 -c fitFluxCal:ignoredBroadbandFilters=[] -c fitFluxCal:fitFocalPlane.fitPrecisely=False --extend-run

###############
## run drpQa ##
###############

echo "Start DRP QA..."
date

## detectorMapQa ##

pipetask --long-log --log-level pfs=INFO run --register-dataset-types -b /work/datastore --instrument lsst.obs.pfs.PrimeFocusSpectrograph -i OUTPUT_TEMPLATE/VISITS_TEMPLATE -o OUTPUT_TEMPLATE/VISITS_TEMPLATE.dmQa -p $DRP_QA_DIR/pipelines/drpQA.yaml#dmResiduals,dmCombinedResiduals -d "visit = VISITS_TEMPLATE" --fail-fast -j 12 --rebase

find /work/datastore/OUTPUT_TEMPLATE/VISITS_TEMPLATE.dmQa -name "*.pdf" | xargs -I {} rsync -av {} WORKDIR_TEMPLATE/figures/drpQa/detectorMapQa
find /work/datastore/OUTPUT_TEMPLATE/VISITS_TEMPLATE.dmQa -name "*.png" | xargs -I {} rsync -av {} WORKDIR_TEMPLATE/figures/drpQa/detectorMapQa

## skySubtractionQa ##

# (it takes time...) skip this process for now 
##cd WORKDIR_TEMPLATE/figures/drpQa/skySubtractionQa
##WORKDIR_TEMPLATE/src/drp_qa/python/pfs/drp/qa/skySubtraction/generatePlots.py --collection OUTPUT_TEMPLATE/VISITS_TEMPLATE --visit VISITS_TEMPLATE --spectrograph 1 --arms b r n
##WORKDIR_TEMPLATE/src/drp_qa/python/pfs/drp/qa/skySubtraction/generatePlots.py --collection OUTPUT_TEMPLATE/VISITS_TEMPLATE --visit VISITS_TEMPLATE --spectrograph 2 --arms b r n
##WORKDIR_TEMPLATE/src/drp_qa/python/pfs/drp/qa/skySubtraction/generatePlots.py --collection OUTPUT_TEMPLATE/VISITS_TEMPLATE --visit VISITS_TEMPLATE --spectrograph 3 --arms b r n
##WORKDIR_TEMPLATE/src/drp_qa/python/pfs/drp/qa/skySubtraction/generatePlots.py --collection OUTPUT_TEMPLATE/VISITS_TEMPLATE --visit VISITS_TEMPLATE --spectrograph 4 --arms b r n

## fluxCalQa (with fluxCal) ##

cd WORKDIR_TEMPLATE/figures/drpQa/fluxCalQa
WORKDIR_TEMPLATE/src/drp_qa/python/pfs/drp/qa/fluxCal/fluxCalQA.py OUTPUT_TEMPLATE/VISITS_TEMPLATE VISITS_TEMPLATE --saveData

##########################
## clean-up collections ##
##########################

butlerCleanRun.py /work/datastore OUTPUT_TEMPLATE/VISITS_TEMPLATE/* postISRCCD

echo "Finished!"
date
