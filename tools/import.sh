#!/bin/bash
SRCDIR='../nlp100-2025'
rsync -avh $SRCDIR/data ./
cp $SRCDIR/tools/build.sh ./tools/
cp $SRCDIR/tools/modify.py ./tools/
cp $SRCDIR/tools/upload.sh ./tools/
cp -r $SRCDIR/ja ./
jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags answer --TagRemovePreprocessor.remove_all_outputs_tags answer --TagRemovePreprocessor.remove_input_tags answer --to notebook --inplace ja/*.ipynb
