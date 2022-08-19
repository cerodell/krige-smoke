# jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-data.py
# mv /Users/rodell/krige-smoke/scripts/comps-data.ipynb /Users/rodell/krige-smoke/docs/source/

# jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-ok.py
# mv /Users/rodell/krige-smoke/scripts/comps-ok.ipynb /Users/rodell/krige-smoke/docs/source/


jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-uk-bsp.py
mv /Users/rodell/krige-smoke/scripts/comps-uk-bsp.ipynb /Users/rodell/krige-smoke/docs/source/

jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-rk-dem.py
mv /Users/rodell/krige-smoke/scripts/comps-rk-dem.ipynb /Users/rodell/krige-smoke/docs/source/

jupytext --to notebook /Users/rodell/krige-smoke/scripts/comps-ver.py
mv /Users/rodell/krige-smoke/scripts/comps-ver.ipynb /Users/rodell/krige-smoke/docs/source/

cd /Users/rodell/krige-smoke/docs

# make clean
make html
