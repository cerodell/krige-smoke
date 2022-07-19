jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps.py
mv /Users/rodell/krige-smoke/scripts/comps.ipynb /Users/rodell/krige-smoke/docs/source/

jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-ok.py
mv /Users/rodell/krige-smoke/scripts/comps-ok.ipynb /Users/rodell/krige-smoke/docs/source/


jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-uk.py
mv /Users/rodell/krige-smoke/scripts/comps-uk.ipynb /Users/rodell/krige-smoke/docs/source/

cd /Users/rodell/krige-smoke/docs

make clean
make html
