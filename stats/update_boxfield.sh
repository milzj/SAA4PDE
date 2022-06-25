rm -f boxfield.py
wget https://raw.githubusercontent.com/hplgit/fenics-tutorial/master/src/vol1/python/boxfield.py
perl -pi -e "s/xrange/range/g" boxfield.py
perl -pi -e "s/\.array/\.get_local/g" boxfield.py
python update_boxfield.py
